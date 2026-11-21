"""Monitoring: Prometheus metrics (scraped at /metrics) + an in-process rolling window for the UI / JSON summary.

Signals:
* service health  — request count, errors, latency per endpoint;
* model behaviour — predicted class distribution, confidence, share of low-confidence answers,
  degraded (fallback) generations, RAG context copies;
* data drift      — out-of-vocabulary rate of incoming ingredients vs. the validation baseline from the manifest.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from prometheus_client import CollectorRegistry, Gauge, Histogram, Info
from prometheus_client import Counter as PCounter

LAT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        data.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})


@dataclass
class Event:
    ts: float
    endpoint: str
    latency_s: float
    ok: bool
    label: str | None = None
    confidence: float | None = None
    oov: float | None = None
    degraded: bool = False
    context_copy: bool = False


@dataclass
class Monitor:
    window: int = 500
    low_confidence: float = 0.4
    oov_baseline: float | None = None
    drift_margin: float = 0.15
    prediction_log: Path | None = None
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        r = self.registry
        self.requests = PCounter("menu_requests_total", "API requests", ["endpoint", "status"], registry=r)
        self.latency = Histogram(
            "menu_request_latency_seconds", "Request latency", ["endpoint"], buckets=LAT_BUCKETS, registry=r
        )
        self.inference = Histogram(
            "menu_model_latency_seconds",
            "Model inference latency",
            ["model"],
            buckets=LAT_BUCKETS,
            registry=r,
        )
        self.predictions = PCounter("menu_predicted_class_total", "Predicted dish types", ["label"], registry=r)
        self.confidence = Histogram(
            "menu_prediction_confidence",
            "Top-1 probability",
            buckets=[i / 10 for i in range(1, 11)],
            registry=r,
        )
        self.low_conf = PCounter("menu_low_confidence_total", "Predictions below the confidence threshold", registry=r)
        self.oov = Histogram(
            "menu_input_oov_rate",
            "Share of unknown input words",
            buckets=[i / 10 for i in range(0, 11)],
            registry=r,
        )
        self.degraded = PCounter("menu_degraded_generations_total", "Generations served by the fallback", registry=r)
        self.ctx_copy = PCounter(
            "menu_rag_context_copies_total", "Generated titles copied from retrieved recipes", registry=r
        )
        self.drift = Gauge("menu_drift_alert", "1 if rolling OOV rate exceeds baseline + margin", registry=r)
        self.models_loaded = Gauge("menu_model_loaded", "Model component loaded", ["component"], registry=r)
        self.info = Info("menu_build", "Bundle and package versions", registry=r)
        self._events: deque[Event] = deque(maxlen=self.window)
        self._lock = threading.Lock()
        self.started = time.time()

    def record(self, ev: Event) -> None:
        self.requests.labels(ev.endpoint, "ok" if ev.ok else "error").inc()
        self.latency.labels(ev.endpoint).observe(ev.latency_s)
        if ev.label is not None:
            self.predictions.labels(ev.label).inc()
        if ev.confidence is not None:
            self.confidence.observe(ev.confidence)
            if ev.confidence < self.low_confidence:
                self.low_conf.inc()
        if ev.oov is not None:
            self.oov.observe(ev.oov)
        if ev.degraded:
            self.degraded.inc()
        if ev.context_copy:
            self.ctx_copy.inc()
        with self._lock:
            self._events.append(ev)
        self.drift.set(1.0 if self.summary()["drift_alert"] else 0.0)
        if self.prediction_log:
            with open(self.prediction_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev.__dict__) + "\n")

    def summary(self) -> dict:
        with self._lock:
            evs = list(self._events)
        lat = np.array([e.latency_s for e in evs]) if evs else np.array([0.0])
        conf = [e.confidence for e in evs if e.confidence is not None]
        oov = [e.oov for e in evs if e.oov is not None]
        oov_mean = float(np.mean(oov)) if oov else None
        drift = bool(
            self.oov_baseline is not None
            and oov_mean is not None
            and len(oov) >= 20
            and oov_mean > self.oov_baseline + self.drift_margin
        )
        by_ep = Counter(e.endpoint for e in evs)
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "window_requests": len(evs),
            "requests_by_endpoint": dict(by_ep),
            "error_rate": float(np.mean([not e.ok for e in evs])) if evs else 0.0,
            "latency_p50_ms": round(float(np.percentile(lat, 50)) * 1000, 1),
            "latency_p95_ms": round(float(np.percentile(lat, 95)) * 1000, 1),
            "class_distribution": dict(Counter(e.label for e in evs if e.label)),
            "mean_confidence": round(float(np.mean(conf)), 3) if conf else None,
            "low_confidence_share": round(float(np.mean([c < self.low_confidence for c in conf])), 3) if conf else None,
            "oov_rate_mean": round(oov_mean, 3) if oov_mean is not None else None,
            "oov_rate_baseline": round(self.oov_baseline, 3) if self.oov_baseline is not None else None,
            "drift_alert": drift,
            "degraded_share": round(float(np.mean([e.degraded for e in evs])), 3) if evs else 0.0,
        }

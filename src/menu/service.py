"""MenuService: loads the versioned bundle once and serves every use case (API and UI call the same object)."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from menu.bundle import ensure_bundle, read_manifest, verify_bundle
from menu.models.classifier import DishClassifier
from menu.models.generator import (
    GeneratorSpec,
    QwenGenerator,
    RetrievalFallbackGenerator,
    TextGenerator,
    adapter_dirs,
)
from menu.monitoring import Event, Monitor, log_event
from menu.rag.prompts import chat_messages, clean_title, title_messages
from menu.rag.retriever import Hit, RecipeRetriever
from menu.schemas import (
    ChatOut,
    ClassifyOut,
    ClassScore,
    GenerateOut,
    MenuOut,
    Meta,
    RecipeHit,
    SearchOut,
)
from menu.settings import Settings
from menu.text import to_clean, to_raw

log = logging.getLogger("menu.service")

MODES: dict[str, tuple[str | None, int]] = {  # mode -> (adapter, default rag_k)
    "zero_shot": (None, 0),
    "zero_shot_rag": (None, 3),
    "lora": ("lora", 0),
    "lora_rag": ("lora_rag", 3),
}


def _hit(h: Hit) -> RecipeHit:
    return RecipeHit(**h.as_dict())


@dataclass
class MenuService:
    settings: Settings
    monitor: Monitor = field(init=False)
    manifest: dict = field(default_factory=dict)
    classifier: DishClassifier | None = None
    retriever: RecipeRetriever | None = None
    generator: TextGenerator | None = None
    llm_ready: bool = False
    llm_error: str | None = None
    dense_error: str | None = None
    integrity_errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        s = self.settings
        self.monitor = Monitor(
            window=s.monitoring_window,
            low_confidence=s.low_confidence,
            drift_margin=s.drift_oov_margin,
            prediction_log=s.prediction_log,
        )

    # ------------------------------------------------------------------ loading
    def load(self, background_llm: bool = True) -> MenuService:
        s = self.settings
        root = ensure_bundle(s.artifacts_dir, s.hf_repo, s.hf_revision, s.hf_repo_type)
        self.root = root
        self.manifest = read_manifest(root)
        self.integrity_errors = verify_bundle(root)
        if self.integrity_errors:
            log.warning("bundle checksum mismatch: %s", self.integrity_errors)
        self.classifier = DishClassifier.load(root / "classifier.joblib")
        self.retriever = RecipeRetriever.load(root / "retriever.joblib")
        self.monitor.oov_baseline = self.manifest.get("monitoring_baseline", {}).get("oov_rate_validation")
        self.monitor.info.info({"bundle": self.version, "package": self.manifest.get("package_version", "?")})
        self.monitor.models_loaded.labels("classifier").set(1)
        self.monitor.models_loaded.labels("retriever").set(1)
        self.generator = RetrievalFallbackGenerator()
        if s.enable_llm:
            if background_llm:
                threading.Thread(target=self._load_llm, name="llm-loader", daemon=True).start()
            else:
                self._load_llm()
        else:
            self.llm_error = "disabled by MENU_ENABLE_LLM=false"
        log_event(log, "bundle loaded", bundle=self.version, llm=s.enable_llm)
        return self

    def _load_llm(self) -> None:
        s, g = self.settings, self.manifest.get("generator", {})
        t0 = time.perf_counter()
        if s.enable_dense and self.retriever is not None and self.retriever.dense is not None:
            try:  # optional: chat/search fall back to BM25 without it, so it must not block the LLM
                from menu.rag.retriever import sentence_encoder

                self.retriever.attach_encoder(sentence_encoder(self.manifest["retriever"]["dense_model"]))
                self.monitor.models_loaded.labels("dense_encoder").set(1)
            except Exception as exc:
                self.dense_error = f"{type(exc).__name__}: {exc}"
                log.exception("dense encoder failed to load; search uses BM25 only")
        try:
            spec = GeneratorSpec(
                base_model=g.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct"),
                base_revision=g.get("base_revision"),
                adapters=adapter_dirs(self.root / "generator", g.get("adapters", [])),
                device=s.llm_device,
                num_threads=s.llm_threads,
            )
            self.generator = QwenGenerator(spec)
            self.llm_ready = True
            self.monitor.models_loaded.labels("generator").set(1)
            log_event(log, "llm loaded", seconds=round(time.perf_counter() - t0, 1), adapters=spec.adapters)
        except Exception as exc:  # keep serving classification/search in degraded mode
            self.llm_error = f"{type(exc).__name__}: {exc}"
            log.exception("LLM failed to load; generation falls back to retrieval")

    # ------------------------------------------------------------------ info
    @property
    def version(self) -> str:
        return str(self.manifest.get("bundle_version", "unknown"))

    @property
    def final_variant(self) -> dict:
        return dict(self.manifest.get("generator", {}).get("final_variant", {"adapter": "lora", "rag_k": 0}))

    def health(self) -> dict[str, Any]:
        comps = {
            "classifier": self.classifier is not None,
            "retriever": self.retriever is not None,
            "generator_llm": self.llm_ready,
            "bundle_integrity": not self.integrity_errors,
        }
        if not (comps["classifier"] and comps["retriever"]) or (
            self.settings.enable_llm and not self.llm_ready and self.llm_error is None
        ):
            status = "loading"
        else:
            status = "ok" if all(comps.values()) else "degraded"
        errors = {k: v for k, v in {"generator_llm": self.llm_error, "dense_encoder": self.dense_error}.items() if v}
        return {"status": status, "components": comps, "bundle_version": self.version or None, "errors": errors}

    def _meta(self, t0: float, warnings: list[str] | None = None) -> Meta:
        return Meta(
            model_version=self.version,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            warnings=warnings or [],
        )

    # ------------------------------------------------------------------ use cases
    def classify(self, items: list[str], top_k: int = 3, _endpoint: str = "classify") -> ClassifyOut:
        assert self.classifier is not None
        t0 = time.perf_counter()
        ok, out = False, None
        try:
            text = to_raw(items)
            t1 = time.perf_counter()
            pred = self.classifier.predict_one(text, top_k=top_k)
            self.monitor.inference.labels("classifier").observe(time.perf_counter() - t1)
            oov = self.classifier.oov_rate(text)
            warnings = []
            if oov > 0.5:
                warnings.append(f"{oov:.0%} of the input words are unknown to the model (English ingredients expected)")
            low = pred.confidence < self.settings.low_confidence
            if low:
                warnings.append("low confidence: the dish type is ambiguous")
            out = ClassifyOut(
                dish_type=pred.label,
                confidence=round(pred.confidence, 4),
                top=[ClassScore(label=lb, probability=round(p, 4)) for lb, p in pred.top],
                low_confidence=low,
                normalized_ingredients=items,
                meta=self._meta(t0, warnings),
            )
            ok = True
            return out
        finally:
            self.monitor.record(
                Event(
                    time.time(),
                    _endpoint,
                    time.perf_counter() - t0,
                    ok,
                    label=out.dish_type if out else None,
                    confidence=out.confidence if out else None,
                    oov=self.classifier.oov_rate(to_raw(items)) if ok else None,
                )
            )

    def generate(
        self, items: list[str], mode: str = "auto", rag_k: int | None = None, _endpoint: str = "generate"
    ) -> GenerateOut:
        assert self.retriever is not None and self.generator is not None
        t0 = time.perf_counter()
        warnings: list[str] = []
        ok, out = False, None
        try:
            if mode == "auto":
                fv = self.final_variant
                adapter, k = fv.get("adapter"), int(fv.get("rag_k", 0))
                mode_name = next((m for m, (a, kk) in MODES.items() if a == adapter and bool(kk) == bool(k)), "lora")
            else:
                adapter, k = MODES[mode]
                mode_name = mode
            if rag_k is not None and mode_name.endswith("rag"):
                k = rag_k
            if adapter and self.llm_ready and adapter not in self.generator.adapters:
                warnings.append(f"adapter {adapter!r} is not in this bundle; using the base model")
                adapter = None
            degraded = not self.llm_ready
            ing = to_clean(items) or "; ".join(items)
            hits = self.retriever.search(ing, k=max(k, 1 if degraded else 0), mode="sparse") if (k or degraded) else []
            context = hits[:k] if k else []
            messages = title_messages(ing, hits if degraded else context)
            t1 = time.perf_counter()
            raw = self.generator.generate(
                [messages],
                adapter=adapter,
                decoding=self.final_variant.get("decoding", "beam4"),
                max_new_tokens=self.manifest.get("generator", {}).get("max_new_tokens", 24),
            )
            self.monitor.inference.labels("generator").observe(time.perf_counter() - t1)
            title, valid = clean_title(raw[0] if raw else "")
            if degraded:
                warnings.append("LLM unavailable: the title of the most similar recipe is returned instead")
            norm = re.sub(r"\W+", " ", title.lower()).strip()
            copy = any(norm == re.sub(r"\W+", " ", h.title.lower()).strip() for h in context)
            out = GenerateOut(
                title=title,
                valid=valid,
                mode=mode_name if not degraded else "retrieval_fallback",
                degraded=degraded,
                context=[_hit(h) for h in context],
                context_copy=copy,
                meta=self._meta(t0, warnings),
            )
            ok = True
            return out
        finally:
            self.monitor.record(
                Event(
                    time.time(),
                    _endpoint,
                    time.perf_counter() - t0,
                    ok,
                    degraded=bool(out and out.degraded),
                    context_copy=bool(out and out.context_copy),
                )
            )

    def menu(self, items: list[str], mode: str = "auto") -> MenuOut:
        return MenuOut(
            classification=self.classify(items, _endpoint="menu"),
            generation=self.generate(items, mode=mode, _endpoint="menu"),
        )

    def search(self, query: str, k: int = 5) -> SearchOut:
        assert self.retriever is not None
        t0 = time.perf_counter()
        ok = False
        try:
            hits = self.retriever.search(query, k=k, mode="hybrid", title_weight=0.5)
            ok = True
            return SearchOut(hits=[_hit(h) for h in hits], meta=self._meta(t0))
        finally:
            self.monitor.record(Event(time.time(), "search", time.perf_counter() - t0, ok))

    def chat(self, message: str, k: int = 4) -> ChatOut:
        assert self.retriever is not None and self.classifier is not None and self.generator is not None
        t0 = time.perf_counter()
        ok = False
        try:
            hits = self.retriever.search(message, k=k, mode="hybrid", title_weight=0.5)
            dish = self.classifier.predict_one(message).label if hits else None
            warnings: list[str] = []
            if not hits:
                answer, grounded = (
                    "I could not find matching recipes. Try listing a few ingredients in English.",
                    False,
                )
            elif self.llm_ready:
                raw = self.generator.generate(
                    [chat_messages(message, hits)],
                    adapter=None,
                    decoding="greedy",
                    max_new_tokens=self.settings.chat_max_new_tokens,
                )
                answer = raw[0].strip()
                grounded = _mentions(answer, hits)
                if not grounded:
                    warnings.append("the answer does not cite a retrieved recipe; sources are listed below")
            else:
                answer, grounded = _template_answer(message, hits), True
            out = ChatOut(
                answer=answer,
                sources=[_hit(h) for h in hits],
                suggested_dish_type=dish,
                grounded=grounded,
                degraded=not self.llm_ready,
                meta=self._meta(t0, warnings),
            )
            ok = True
            return out
        finally:
            self.monitor.record(Event(time.time(), "chat", time.perf_counter() - t0, ok))


def _mentions(answer: str, hits: list[Hit]) -> bool:
    a = answer.lower()
    return any(h.title.lower() in a or f"[{i + 1}]" in a for i, h in enumerate(hits))


def _template_answer(question: str, hits: list[Hit]) -> str:
    q = set(re.findall(r"[a-z]+", question.lower()))
    best = hits[0]
    have = [x for x in best.ingredients.split("; ") if set(x.split()) & q]
    missing = [x for x in best.ingredients.split("; ") if not set(x.split()) & q][:5]
    parts = [f"Try “{best.title}”" + (f" ({best.course.lower()})" if best.course else "") + "."]
    if have:
        parts.append("It uses " + ", ".join(have[:4]) + ".")
    if missing:
        parts.append("You would also need " + ", ".join(missing) + ".")
    if len(hits) > 1:
        parts.append("Alternatives: " + "; ".join(h.title for h in hits[1:3]) + ".")
    return " ".join(parts)

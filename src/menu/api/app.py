"""FastAPI application: REST API (/v1), health probes, Prometheus metrics and the Gradio UI mounted at /ui."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from menu import __version__
from menu.monitoring import log_event, setup_logging
from menu.schemas import (
    ChatIn,
    ChatOut,
    ClassifyIn,
    ClassifyOut,
    GenerateIn,
    GenerateOut,
    Health,
    IngredientsIn,
    MenuOut,
    ModelInfo,
    SearchIn,
    SearchOut,
)
from menu.service import MenuService
from menu.settings import Settings, get_settings

log = logging.getLogger("menu.api")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

DESCRIPTION = """
**Menu from Neural Networks** — from a list of ingredients the system predicts the **dish type**
(TF-IDF + LinearSVC), invents a **dish name** (Qwen2.5-0.5B-Instruct + LoRA, optionally with retrieved recipes
as context — RAG) and answers **"what can I cook?"** questions grounded in 14.5k Epicurious recipes.

Every response carries `meta.model_version` (the pinned model bundle) and `meta.latency_ms`.
"""


def create_app(settings: Settings | None = None, service: MenuService | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            app.state.service = MenuService(settings).load()
        yield

    app = FastAPI(
        title="Menu from Neural Networks API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.service = service

    def svc(request: Request) -> MenuService:
        s: MenuService | None = request.app.state.service
        if s is None or s.classifier is None:
            raise HTTPException(503, "model bundle is still loading")
        return s

    def auth(key: str | None = Security(api_key_header)) -> None:
        if settings.api_key and key != settings.api_key:
            raise HTTPException(401, "invalid or missing X-API-Key")

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"extra_fields": {"request_id": rid}})
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        response.headers["X-Request-ID"] = rid
        if not request.url.path.startswith(("/ui", "/metrics")):
            log_event(
                log,
                "request",
                request_id=rid,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                ms=round((time.perf_counter() - t0) * 1000, 1),
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        s: MenuService | None = request.app.state.service
        if s is not None:
            s.monitor.requests.labels(request.url.path.rsplit("/", 1)[-1], "invalid").inc()
        return JSONResponse(
            {"detail": jsonable_encoder(exc.errors(), custom_encoder={Exception: str})}, status_code=422
        )

    # ----------------------------------------------------------------- probes & ops
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui" if settings.mount_ui else "/docs")

    @app.get("/health", response_model=Health, tags=["ops"], summary="Liveness/readiness of every component")
    def health(request: Request):
        s: MenuService | None = request.app.state.service
        if s is None:
            return Health(status="loading", components={}, bundle_version=None)
        return Health(**s.health())

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request):
        s = svc(request)
        return Response(generate_latest(s.monitor.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get(
        "/v1/monitoring",
        tags=["ops"],
        summary="Rolling-window monitoring summary (latency, confidence, drift)",
    )
    def monitoring(s: MenuService = Depends(svc)) -> dict:
        return s.monitor.summary()

    @app.get(
        "/v1/model",
        response_model=ModelInfo,
        tags=["model"],
        summary="Pinned model versions and I/O contract",
    )
    def model_info(s: MenuService = Depends(svc)):
        m = s.manifest
        return ModelInfo(
            bundle_version=s.version,
            package_version=m.get("package_version", __version__),
            components={k: m.get(k, {}) for k in ("data", "classifier", "retriever", "generator")},
            final_generator=s.final_variant,
            input_schema=GenerateIn.model_json_schema(),
            output_schema=MenuOut.model_json_schema(),
        )

    # ----------------------------------------------------------------- inference (sync -> threadpool)
    deps = [Depends(auth)]

    @app.post(
        "/v1/classify",
        response_model=ClassifyOut,
        tags=["inference"],
        dependencies=deps,
        summary="Predict the dish type",
    )
    def classify(body: ClassifyIn, s: MenuService = Depends(svc)):
        return s.classify(body.ingredients, top_k=body.top_k)  # type: ignore[arg-type]

    @app.post(
        "/v1/generate",
        response_model=GenerateOut,
        tags=["inference"],
        dependencies=deps,
        summary="Generate a dish name (LLM + LoRA, optional RAG)",
    )
    def generate(body: GenerateIn, s: MenuService = Depends(svc)):
        return s.generate(body.ingredients, mode=body.mode, rag_k=body.rag_k)  # type: ignore[arg-type]

    @app.post(
        "/v1/menu",
        response_model=MenuOut,
        tags=["inference"],
        dependencies=deps,
        summary="Dish type + dish name in one call",
    )
    def menu(body: IngredientsIn, s: MenuService = Depends(svc)):
        return s.menu(body.ingredients)  # type: ignore[arg-type]

    @app.post(
        "/v1/search",
        response_model=SearchOut,
        tags=["rag"],
        dependencies=deps,
        summary="Find similar recipes (BM25 + dense, RRF)",
    )
    def search(body: SearchIn, s: MenuService = Depends(svc)):
        return s.search(body.query, k=body.k)

    @app.post(
        "/v1/chat",
        response_model=ChatOut,
        tags=["rag"],
        dependencies=deps,
        summary="Recipe assistant grounded in retrieved recipes",
    )
    def chat(body: ChatIn, s: MenuService = Depends(svc)):
        return s.chat(body.message, k=body.k)

    if settings.mount_ui:
        import gradio as gr

        from menu.ui.gradio_app import CSS, THEME, build_ui

        app = gr.mount_gradio_app(
            app, build_ui(lambda: app.state.service), path="/ui", theme=THEME, css=CSS, footer_links=["api"]
        )
    return app

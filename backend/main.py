"""FastAPI application.

Run from the repository root:
    uvicorn backend.main:app --reload --port 8000
Interactive API docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import mimetypes
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.config import Settings
from backend.schemas import (ErrorResponse, EvaluationResponse, HealthResponse, ModelsResponse, SearchRequest,
                             SearchResponse, StatsResponse)
from backend.services.evaluation import collect_evaluation, collect_stats
from src.retrieval.search import SearchError, SearchService
from src.utils.common import get_logger

log = get_logger("ncir.api")

ERROR_RESPONSES = {422: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details}})


def create_app(service: Optional[SearchService] = None, settings: Optional[Settings] = None) -> FastAPI:
    """Application factory. Pass ``service`` / ``settings`` in tests; otherwise they are built at startup
    from the environment. Models are loaded ONCE at startup, never per request."""
    cfg = settings or Settings.from_env()
    logging.getLogger("ncir").setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or SearchService.from_artifacts(
            cfg.artifacts_dir, cfg.metadata_path, cfg.images_root, max_top_k=cfg.max_top_k,
            max_query_chars=cfg.max_query_chars)
        svc: SearchService = app.state.service
        for w in svc.startup_warnings:
            log.warning("startup check: %s", w)
        log.info("loaded models: %s", {m["name"]: m["available"] for m in svc.list_models()})
        yield

    app = FastAPI(title="NeuralSeek - Semantic Image Retrieval API", version="1.0.0", lifespan=lifespan,
                  description="Text-to-image retrieval with a joint image-caption embedding space.")
    app.state.settings = cfg
    app.state.service = service  # replaced in lifespan when None
    app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    def svc_of(request: Request) -> SearchService:
        return request.app.state.service

    # ------------------------------------------------------------------ error handling
    @app.exception_handler(SearchError)
    async def search_error_handler(_: Request, exc: SearchError):
        return _error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "problem": e["msg"]} for e in exc.errors()]
        return _error(422, "invalid_request", "The request body is invalid.", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception):
        log.exception("unhandled error: %s", exc)
        return _error(500, "internal_error", "Unexpected server error. See the server log.")

    # ------------------------------------------------------------------ routes
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request, response: Response):
        """Readiness summary. A degraded service returns 503 so hosting health checks detect it."""
        svc = svc_of(request)
        models = svc.list_models()
        avail = [m for m in models if m["available"]]
        response.status_code = 200 if avail else 503
        return HealthResponse(status="ok" if avail else "degraded", models_total=len(models),
                              models_available=len(avail), default_model=svc.default_model,
                              warnings=svc.startup_warnings)

    @app.get("/api/models", response_model=ModelsResponse, tags=["models"])
    def models(request: Request):
        """Models found in the artifacts folder with their architecture, checkpoint and evaluation metadata."""
        svc = svc_of(request)
        return ModelsResponse(default_model=svc.default_model, models=svc.list_models())

    @app.get("/api/stats", response_model=StatsResponse, tags=["data"])
    def stats(request: Request):
        """Dataset statistics (split sizes, caption lengths, vocabulary) and indexed image counts."""
        svc = svc_of(request)
        return collect_stats(cfg.stats_path, cfg.metadata_path, svc.list_models())

    @app.get("/api/evaluation", response_model=EvaluationResponse, tags=["evaluation"])
    def evaluation():
        """Measured metrics and training curves. ``available=false`` until experiments have been run."""
        return collect_evaluation(cfg.reports_dir, cfg.artifacts_dir)

    @app.post("/api/search", response_model=SearchResponse, responses=ERROR_RESPONSES, tags=["search"])
    def search(body: SearchRequest, request: Request):
        """Rank indexed images for a natural-language query (cosine similarity of embeddings)."""
        return svc_of(request).search(body.query, body.top_k, body.model)

    @app.get("/api/images/{image_id}", responses={404: {"model": ErrorResponse}}, tags=["search"])
    def image(image_id: str, request: Request):
        """Serve an indexed image. Only ids present in the dataset metadata resolve; no client path is used."""
        svc = svc_of(request)
        path = svc.image_store.path_for(image_id) if svc.image_store else None
        if path is None:
            return _error(404, "image_not_found", f"No image file for id '{image_id}'.")
        media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media, headers={"Cache-Control": "public, max-age=3600"})

    return app


app = create_app()

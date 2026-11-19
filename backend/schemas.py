"""Pydantic request / response models (these define the API contract used by the frontend)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    # Deeper validation (blank queries, top_k bounds) happens in SearchService so the CLI, tests and API
    # share exactly the same rules and error messages.
    query: str = Field(..., description="Natural-language description, e.g. 'a dog running through snow'.")
    top_k: int = Field(10, strict=True, description="Number of images to return (1 .. NCIR_MAX_TOP_K).")
    model: Optional[str] = Field(None, description="Model name from GET /api/models. Defaults to the first available model.")


class SearchResult(BaseModel):
    rank: int
    image_id: str
    score: float = Field(..., description="Cosine similarity in [-1, 1]. A ranking score, not a probability.")
    image_url: str
    file_name: Optional[str] = None
    captions: list[str] = []
    split: Optional[str] = None
    image_available: bool = True


class SearchResponse(BaseModel):
    query: str
    model: str
    top_k: int
    num_results: int
    unknown_tokens: list[str]
    score_type: str
    score_note: str
    latency_ms: float
    results: list[SearchResult]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    models_total: int
    models_available: int
    default_model: Optional[str]
    warnings: list[str]


class ModelInfo(BaseModel):
    name: str
    type: str
    embedding_dim: Optional[int] = None
    available: bool
    unavailable_reason: Optional[str] = None
    num_images: int = 0
    index_backend: Optional[str] = None
    metadata: dict[str, Any] = {}


class ModelsResponse(BaseModel):
    default_model: Optional[str]
    models: list[ModelInfo]


class StatsResponse(BaseModel):
    available: bool
    message: Optional[str] = None
    dataset: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    indexed_images: dict[str, int] = {}
    synthetic: bool = False


class EvaluationResponse(BaseModel):
    available: bool
    message: Optional[str] = None
    results: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}
    reference: dict[str, Any] = {}
    synthetic_warning: bool = False

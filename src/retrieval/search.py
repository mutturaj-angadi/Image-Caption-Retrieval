"""Query-time retrieval service: validation, encoding, ranking and result assembly.

Scores are cosine similarities of unit-norm embeddings in [-1, 1]. They are rankings, NOT calibrated
probabilities: a score of 0.4 does not mean "40 % relevant".
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.data.pairs import load_records
from src.data.records import ImageRecord
from src.data.text import tokenize
from src.retrieval.index import ImageIndex, IndexError_
from src.retrieval.models import BaseRetriever, ModelArtifactError, load_retriever
from src.utils.common import get_logger, load_json, resolve_path

log = get_logger("ncir.search")

SCORE_NOTE = ("Cosine similarity between the query and image embeddings (higher = closer). "
              "It is a ranking score, not a probability.")


# ---------------------------------------------------------------------- errors
class SearchError(Exception):
    """Base class; carries an HTTP-style status code and a stable machine-readable code."""

    status_code = 400
    code = "search_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidQueryError(SearchError):
    status_code = 422
    code = "invalid_query"


class ModelNotFoundError(SearchError):
    status_code = 404
    code = "model_not_found"


class ModelUnavailableError(SearchError):
    status_code = 503
    code = "model_unavailable"


# ---------------------------------------------------------------------- image store
class ImageStore:
    """Maps image ids to files under a fixed root. Only ids that exist in the metadata are served,
    so user input can never select an arbitrary filesystem path."""

    def __init__(self, images_root: Path, records: list[ImageRecord]):
        self.root = Path(images_root).resolve()
        self.records = {r.image_id: r for r in records}

    def record(self, image_id: str) -> Optional[ImageRecord]:
        return self.records.get(image_id)

    def path_for(self, image_id: str) -> Optional[Path]:
        rec = self.records.get(image_id)
        if rec is None:
            return None
        path = (self.root / rec.file_name).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return None  # metadata tried to escape the root
        return path if path.is_file() else None

    def count_missing(self, ids: list[str]) -> int:
        return sum(1 for i in ids if self.path_for(i) is None)


# ---------------------------------------------------------------------- loaded model
@dataclass
class LoadedModel:
    name: str
    metadata: dict[str, Any]
    retriever: Optional[BaseRetriever] = None
    index: Optional[ImageIndex] = None
    available: bool = False
    reason: Optional[str] = None

    def describe(self) -> dict[str, Any]:
        arch = self.metadata.get("architecture", {})
        return {
            "name": self.name, "type": arch.get("type", "unknown"),
            "embedding_dim": arch.get("embedding_dim"), "available": self.available,
            "unavailable_reason": self.reason, "num_images": self.index.size if self.index else 0,
            "index_backend": self.index.backend if self.index else None,
            "metadata": self.metadata,
        }


class SearchService:
    def __init__(self, models: dict[str, LoadedModel], image_store: Optional[ImageStore] = None,
                 default_model: Optional[str] = None, max_top_k: int = 100, max_query_chars: int = 300,
                 startup_warnings: Optional[list[str]] = None):
        self.models = models
        self.image_store = image_store
        self.max_top_k = max_top_k
        self.max_query_chars = max_query_chars
        self.startup_warnings = startup_warnings or []
        avail = [n for n, m in models.items() if m.available]
        self.default_model = default_model if default_model in avail else (avail[0] if avail else None)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_artifacts(cls, artifacts_dir: str | Path, metadata_path: str | Path, images_root: Optional[str | Path] = None,
                       max_top_k: int = 100, max_query_chars: int = 300, device: str = "cpu",
                       faiss_min_size: int = 20000) -> "SearchService":
        """Load every model folder under ``artifacts_dir`` that contains a ``model_metadata.json``.

        Never raises for missing artifacts: problems are recorded per model (``available=False``) and as
        ``startup_warnings`` so the API can start and report what is wrong.
        """
        warnings: list[str] = []
        art = resolve_path(artifacts_dir)
        store: Optional[ImageStore] = None
        meta_p = resolve_path(metadata_path)
        try:
            records, info = load_records(meta_p)
            root = images_root or info.get("images_root")
            if not root:
                raise ValueError("no images_root in metadata and NCIR_IMAGES_ROOT not set")
            store = ImageStore(resolve_path(root), records)
        except (FileNotFoundError, ValueError, KeyError, OSError):
            warnings.append("dataset metadata unavailable; run scripts/prepare_data.py and configure NCIR_METADATA_PATH")

        models: dict[str, LoadedModel] = {}
        if not art.is_dir():
            warnings.append("artifacts directory not found; train a model and build its index")
        else:
            for run in sorted(p for p in art.iterdir() if p.is_dir() and (p / "model_metadata.json").is_file()):
                try:
                    meta = load_json(run / "model_metadata.json")
                except Exception:  # noqa: BLE001
                    models[run.name] = LoadedModel(run.name, {}, available=False, reason="model metadata is unreadable")
                    continue
                lm = LoadedModel(meta.get("name", run.name), meta)
                try:
                    lm.retriever = load_retriever(run, device=device)
                    lm.index = ImageIndex.load(run / "index", faiss_min_size=faiss_min_size)
                    if lm.index.dim != lm.retriever.embedding_dim:
                        raise ModelArtifactError(
                            f"index dim {lm.index.dim} != model embedding dim {lm.retriever.embedding_dim}; rebuild the index")
                    expected_hash = (meta.get("checkpoint") or {}).get("sha256")
                    indexed_hash = lm.index.meta.get("checkpoint_sha256")
                    if lm.index.meta.get("model") != lm.name:
                        raise ModelArtifactError("index model name does not match model metadata; rebuild the index")
                    if not expected_hash or indexed_hash != expected_hash:
                        raise ModelArtifactError("index checkpoint hash does not match model metadata; rebuild the index")
                    if store is not None:
                        missing = store.count_missing(lm.index.ids)
                        if missing:
                            warnings.append(f"model '{lm.name}': {missing}/{lm.index.size} indexed images have no file on disk")
                    lm.available = True
                except (ModelArtifactError, IndexError_, FileNotFoundError, ValueError) as exc:
                    lm.reason = str(exc)
                    log.warning("model %s unavailable: %s", lm.name, exc)
                except (OSError, RuntimeError):
                    lm.reason = "model artifacts could not be read; check the files and rebuild the index"
                    log.warning("model %s is unavailable because its artifacts could not be read", lm.name)
                models[lm.name] = lm
            if not models:
                warnings.append("no trained models found; run scripts/train.py")
        return cls(models, store, max_top_k=max_top_k, max_query_chars=max_query_chars, startup_warnings=warnings)

    # ------------------------------------------------------------------ queries
    def validate_query(self, query: Any) -> str:
        if not isinstance(query, str):
            raise InvalidQueryError("query must be a string")
        q = " ".join(query.split())
        if not q:
            raise InvalidQueryError("query must not be empty")
        if len(q) > self.max_query_chars:
            raise InvalidQueryError(f"query is too long ({len(q)} > {self.max_query_chars} characters)")
        if not tokenize(q):
            raise InvalidQueryError("query must contain at least one letter or digit")
        return q

    def validate_top_k(self, top_k: Any) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise InvalidQueryError("top_k must be an integer")
        if top_k < 1 or top_k > self.max_top_k:
            raise InvalidQueryError(f"top_k must be between 1 and {self.max_top_k}")
        return top_k

    def get_model(self, name: Optional[str]) -> LoadedModel:
        if not self.models:
            raise ModelUnavailableError("no models are loaded; train a model and build its index first")
        chosen = name or self.default_model
        if chosen is None:
            raise ModelUnavailableError("no model is available: " + "; ".join(
                f"{n}: {m.reason}" for n, m in self.models.items()))
        if chosen not in self.models:
            raise ModelNotFoundError(f"unknown model '{chosen}'. Available: {sorted(self.models)}")
        lm = self.models[chosen]
        if not lm.available or lm.retriever is None or lm.index is None:
            raise ModelUnavailableError(f"model '{chosen}' is not available: {lm.reason or 'unknown reason'}")
        return lm

    def search(self, query: Any, top_k: Any = 10, model: Optional[str] = None) -> dict[str, Any]:
        """Rank the indexed images for a natural-language query."""
        t0 = time.perf_counter()
        q = self.validate_query(query)
        k = self.validate_top_k(top_k)
        lm = self.get_model(model)
        assert lm.retriever is not None and lm.index is not None
        if lm.index.size == 0:
            raise ModelUnavailableError(f"the index of model '{lm.name}' is empty")
        unknown = lm.retriever.unknown_tokens(q)
        if lm.retriever.known_token_count(q) == 0:
            raise InvalidQueryError("none of the words in the query are in the model vocabulary "
                                    f"(unknown: {sorted(set(unknown))[:10]}); try different wording")
        emb = lm.retriever.encode_captions([q])
        scores, pos = lm.index.search(emb[0], k)
        results = []
        for rank, (score, p) in enumerate(zip(scores[0], pos[0]), start=1):
            if p < 0:
                continue
            image_id = lm.index.ids[int(p)]
            rec = self.image_store.record(image_id) if self.image_store else None
            results.append({
                "rank": rank, "image_id": image_id, "score": float(score),
                "image_url": f"/api/images/{image_id}",
                "file_name": rec.file_name if rec else None,
                "captions": list(rec.captions) if rec else [],
                "split": rec.split if rec else None,
                "image_available": bool(self.image_store and self.image_store.path_for(image_id)),
            })
        return {
            "query": q, "model": lm.name, "top_k": k, "num_results": len(results),
            "unknown_tokens": sorted(set(unknown)), "score_type": "cosine_similarity", "score_note": SCORE_NOTE,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "results": results,
        }

    def list_models(self) -> list[dict[str, Any]]:
        return [m.describe() for m in self.models.values()]


# ---------------------------------------------------------------------- module-level convenience
_default_service: Optional[SearchService] = None


def get_default_service() -> SearchService:
    """Lazily build a service from the NCIR_* environment variables (see .env.example)."""
    global _default_service
    if _default_service is None:
        _default_service = SearchService.from_artifacts(
            os.environ.get("NCIR_ARTIFACTS_DIR", "artifacts"),
            os.environ.get("NCIR_METADATA_PATH", "data/processed/metadata.json"),
            os.environ.get("NCIR_IMAGES_ROOT") or None,
            max_top_k=int(os.environ.get("NCIR_MAX_TOP_K", "100")),
            max_query_chars=int(os.environ.get("NCIR_MAX_QUERY_CHARS", "300")))
    return _default_service


def search_images(query: str, top_k: int = 10, model: Optional[str] = None,
                  service: Optional[SearchService] = None) -> dict[str, Any]:
    """Search for images matching ``query``.

    Raises :class:`SearchError` subclasses (never crashes) for empty queries, invalid ``top_k``,
    unknown / unavailable models and empty indexes. Missing image files are reported per result via
    ``image_available`` instead of failing the whole search.
    """
    return (service or get_default_service()).search(query, top_k, model)

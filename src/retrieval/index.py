"""Nearest-neighbour index over L2-normalised image embeddings.

Similarity is the inner product of unit vectors, i.e. cosine similarity. Two interchangeable back-ends
return identical results:

* ``exact`` - NumPy matrix product + top-k (always available; best for small corpora)
* ``faiss`` - ``faiss.IndexFlatIP`` (exact too, but faster for large corpora; optional dependency)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.evaluation.metrics import l2_normalize
from src.utils.common import faiss_available, load_json, save_json


class IndexError_(RuntimeError):
    """Raised for invalid index operations (named with a trailing underscore to avoid the builtin)."""


class ImageIndex:
    def __init__(self, embeddings: np.ndarray, ids: Sequence[str], prefer_faiss: bool = True,
                 faiss_min_size: int = 20000, meta: Optional[dict] = None):
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise IndexError_("embeddings must be a 2-D array [N, D]")
        if emb.shape[0] != len(ids):
            raise IndexError_(f"{emb.shape[0]} embeddings but {len(ids)} ids")
        if len(set(ids)) != len(ids):
            raise IndexError_("image ids must be unique")
        if not np.isfinite(emb).all():
            raise IndexError_("embeddings contain NaN or Inf")
        self.embeddings = l2_normalize(emb)
        self.ids = [str(i) for i in ids]
        self.prefer_faiss = prefer_faiss
        self.faiss_min_size = faiss_min_size
        self.meta = dict(meta or {})
        self._faiss = None
        self._rebuild_backend()

    # ------------------------------------------------------------------ properties
    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "exact"

    # ------------------------------------------------------------------ backend
    def _rebuild_backend(self) -> None:
        self._faiss = None
        if self.prefer_faiss and self.size >= self.faiss_min_size and faiss_available():
            import faiss

            index = faiss.IndexFlatIP(self.dim)
            index.add(np.ascontiguousarray(self.embeddings))
            self._faiss = index

    # ------------------------------------------------------------------ search
    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, positions)`` of the top-``k`` images for each query row (descending score)."""
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.shape[1] != self.dim:
            raise IndexError_(f"query dimension {q.shape[1]} does not match index dimension {self.dim}")
        if self.size == 0:
            raise IndexError_("the index is empty")
        if not np.isfinite(q).all():
            raise IndexError_("query embedding contains NaN or Inf")
        q = l2_normalize(q)
        k = max(1, min(int(k), self.size))
        if self._faiss is not None:
            scores, pos = self._faiss.search(np.ascontiguousarray(q), k)
            return scores, pos
        sim = q @ self.embeddings.T
        if k < self.size:
            part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(self.size), (sim.shape[0], 1))
        part_scores = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        return np.take_along_axis(part_scores, order, axis=1), np.take_along_axis(part, order, axis=1)

    # ------------------------------------------------------------------ updates
    def add(self, embeddings: np.ndarray, ids: Sequence[str]) -> None:
        """Append new images and rebuild the backend (use after indexing additional images)."""
        new = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        if new.shape[1] != self.dim:
            raise IndexError_(f"new embeddings have dimension {new.shape[1]}, index has {self.dim}")
        clash = set(map(str, ids)) & set(self.ids)
        if clash:
            raise IndexError_(f"ids already in the index: {sorted(clash)[:5]}")
        if len(set(ids)) != len(ids) or new.shape[0] != len(ids):
            raise IndexError_("new ids must be unique and match the number of embeddings")
        self.embeddings = np.vstack([self.embeddings, new])
        self.ids.extend(str(i) for i in ids)
        self._rebuild_backend()

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".npy.tmp")
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                np.save(f, self.embeddings, allow_pickle=False)
            os.replace(tmp, d / "embeddings.npy")
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        save_json(self.ids, d / "ids.json")
        save_json({**self.meta, "size": self.size, "dim": self.dim, "backend": self.backend,
                   "format": "ncir-index-v1"}, d / "index_meta.json")

    @classmethod
    def load(cls, directory: str | Path, prefer_faiss: bool = True, faiss_min_size: int = 20000) -> "ImageIndex":
        d = Path(directory)
        for name in ("embeddings.npy", "ids.json", "index_meta.json"):
            if not (d / name).is_file():
                raise FileNotFoundError(f"index file missing: {name} (run scripts/build_index.py)")
        meta = load_json(d / "index_meta.json")
        if not isinstance(meta, dict) or meta.get("format") != "ncir-index-v1":
            raise IndexError_("unsupported or malformed index metadata")
        embeddings = np.load(d / "embeddings.npy", allow_pickle=False)
        ids = load_json(d / "ids.json")
        if not isinstance(ids, list) or meta.get("size") != len(ids) or meta.get("dim") != (embeddings.shape[1] if embeddings.ndim == 2 else None):
            raise IndexError_("index metadata does not match embeddings and ids; rebuild the index")
        return cls(embeddings, ids, prefer_faiss, faiss_min_size, meta)

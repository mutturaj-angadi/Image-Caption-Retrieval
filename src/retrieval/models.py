"""Inference-time wrappers that turn trained artifacts into text/image encoders.

A *retriever* exposes the same three operations for every model type, so evaluation, index building
and the web search all share one code path (and therefore cannot disagree on preprocessing):

* ``encode_captions(list[str])``        -> unit-norm float32 [B, D]
* ``encode_image_features(np.ndarray)`` -> unit-norm float32 [N, D]
* ``unknown_tokens(caption)``           -> query tokens that are not in the training vocabulary
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from src.data.text import Vocabulary, mean_word_vectors, tokenize
from src.evaluation.metrics import l2_normalize
from src.utils.common import load_json, sha256_file


class ModelArtifactError(RuntimeError):
    """A model's files are missing or inconsistent."""


def _validate_checkpoint_hash(run_dir: Path, metadata: dict[str, Any], checkpoint_name: str) -> None:
    """Reject checkpoints that no longer match the model metadata recorded at training time."""
    checkpoint = run_dir / checkpoint_name
    recorded = (metadata.get("checkpoint") or {}).get("sha256")
    if not isinstance(recorded, str) or not recorded:
        raise ModelArtifactError("checkpoint hash is missing from model metadata; retrain the model")
    if sha256_file(checkpoint) != recorded:
        raise ModelArtifactError("checkpoint hash does not match model metadata; retrain or restore the checkpoint")


class BaseRetriever:
    name: str
    kind: str
    embedding_dim: int
    image_feature_dim: int
    metadata: dict[str, Any]
    vocab: Vocabulary

    def unknown_tokens(self, caption: str) -> list[str]:
        return self.vocab.unknown_tokens(tokenize(caption))

    def known_token_count(self, caption: str) -> int:
        toks = tokenize(caption)
        return len(toks) - len(self.vocab.unknown_tokens(toks))

    def encode_captions(self, captions: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def encode_image_features(self, feats: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class BaselineRetriever(BaseRetriever):
    """Mean word vector -> linear regression -> predicted fc7 (compared with raw fc7 by cosine)."""

    kind = "baseline"

    def __init__(self, run_dir: Path, metadata: dict[str, Any]):
        from src.models.baseline import LinearRegressionBaseline

        npz, vocab_path = run_dir / "baseline.npz", run_dir / "vocab.json"
        for p in (npz, vocab_path):
            if not p.is_file():
                raise ModelArtifactError(f"missing model file: {p.name}")
        _validate_checkpoint_hash(run_dir, metadata, npz.name)
        self.model, wv = LinearRegressionBaseline.load(npz)
        if wv is None:
            raise ModelArtifactError(f"{npz} does not contain word vectors")
        self.word_vectors = wv
        self.vocab = Vocabulary.load(vocab_path)
        if wv.shape[0] != len(self.vocab):
            raise ModelArtifactError("word vectors and vocabulary sizes differ")
        self.name = metadata.get("name", run_dir.name)
        self.metadata = metadata
        self.image_feature_dim = int(self.model.W.shape[1])
        self.embedding_dim = self.image_feature_dim  # the shared space IS the fc7 space

    def encode_captions(self, captions: Sequence[str]) -> np.ndarray:
        feats, _ = mean_word_vectors(list(captions), self.vocab, self.word_vectors)
        return l2_normalize(self.model.predict(feats))

    def encode_image_features(self, feats: np.ndarray) -> np.ndarray:
        return l2_normalize(np.asarray(feats, dtype=np.float32))


class NeuralRetriever(BaseRetriever):
    """GRU / LSTM joint embedding model loaded from ``best.pt`` (requires PyTorch)."""

    kind = "neural"

    def __init__(self, run_dir: Path, metadata: dict[str, Any], device: str = "cpu", max_len: Optional[int] = None):
        try:
            import torch
        except ImportError as exc:
            raise ModelArtifactError("PyTorch is not installed, so neural models cannot be loaded") from exc
        from src.models.joint_model import load_checkpoint, model_from_checkpoint

        ckpt_path, vocab_path = run_dir / "best.pt", run_dir / "vocab.json"
        for p in (ckpt_path, vocab_path):
            if not p.is_file():
                raise ModelArtifactError(f"missing model file: {p.name}")
        _validate_checkpoint_hash(run_dir, metadata, ckpt_path.name)
        self._torch = torch
        ckpt = load_checkpoint(ckpt_path)
        self.device = torch.device(device)
        self.model = model_from_checkpoint(ckpt).to(self.device)
        self.vocab = Vocabulary.load(vocab_path)
        if int(ckpt["vocab_size"]) != len(self.vocab):
            raise ModelArtifactError("checkpoint vocabulary size does not match vocab.json")
        self.max_len = int(max_len or ckpt["config"].get("data", {}).get("max_caption_len", 32))
        self.name = metadata.get("name", run_dir.name)
        self.metadata = metadata
        self.image_feature_dim = int(ckpt["image_dim"])
        self.embedding_dim = int(ckpt["config"]["model"].get("embedding_dim", 1024))

    def encode_captions(self, captions: Sequence[str], batch_size: int = 512) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.inference_mode():
            for s in range(0, len(captions), batch_size):
                ids, lens = self.vocab.encode_batch(list(captions[s:s + batch_size]), max_len=self.max_len)
                emb = self.model.encode_text(torch.from_numpy(ids.astype(np.int64)).to(self.device),
                                             torch.from_numpy(lens))
                out.append(emb.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.embedding_dim), dtype=np.float32)

    def encode_image_features(self, feats: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.inference_mode():
            for s in range(0, feats.shape[0], batch_size):
                x = torch.from_numpy(np.ascontiguousarray(feats[s:s + batch_size], dtype=np.float32)).to(self.device)
                out.append(self.model.encode_image(x).cpu().numpy())
        return np.concatenate(out, axis=0)


def read_metadata(run_dir: str | Path) -> dict[str, Any]:
    p = Path(run_dir) / "model_metadata.json"
    if not p.is_file():
        raise ModelArtifactError("model_metadata.json is missing; train the model first (scripts/train.py)")
    return load_json(p)


def load_retriever(run_dir: str | Path, device: str = "cpu") -> BaseRetriever:
    """Load the retriever for ``run_dir`` based on its ``model_metadata.json``."""
    run = Path(run_dir)
    meta = read_metadata(run)
    kind = meta.get("architecture", {}).get("type")
    if kind == "baseline":
        return BaselineRetriever(run, meta)
    if kind in ("gru", "lstm"):
        return NeuralRetriever(run, meta, device=device)
    raise ModelArtifactError(f"unknown model type {kind!r}; check model_metadata.json")

"""Joint image-text embedding model and checkpoint helpers. Requires PyTorch."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.text_encoder import TextEncoder

CKPT_VERSION = "ncir-ckpt-v1"


class JointEmbeddingModel(nn.Module):
    """Maps image features (fc7) and captions into one shared, L2-normalised space.

    ``encode_image``: fc7 [B, F] -> linear -> normalise -> [B, D]
    ``encode_text``:  (ids, lengths) -> TextEncoder -> [B, D]
    Cosine similarity between the two embeddings is the retrieval score.
    """

    def __init__(self, image_dim: int, embedding_dim: int, text_encoder: TextEncoder):
        super().__init__()
        self.image_dim = image_dim
        self.embedding_dim = embedding_dim
        self.image_proj = nn.Linear(image_dim, embedding_dim)
        self.text_encoder = text_encoder

    def encode_image(self, feats: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_proj(feats), dim=-1)

    def encode_text(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(ids, lengths)

    def forward(self, feats: torch.Tensor, ids: torch.Tensor, lengths: torch.Tensor):
        return self.encode_image(feats), self.encode_text(ids, lengths)


def build_joint_model(model_cfg: Mapping[str, Any], image_dim: int, vocab_size: int,
                      embedding_matrix: Optional[np.ndarray]) -> JointEmbeddingModel:
    """Create the model from the ``model`` section of a config."""
    word_dim = int(model_cfg.get("word_dim", 300))
    encoder = TextEncoder(
        vocab_size=vocab_size, word_dim=word_dim, hidden_dim=int(model_cfg.get("hidden_dim", 1024)),
        embedding_dim=int(model_cfg.get("embedding_dim", 1024)), rnn_type=str(model_cfg["type"]),
        num_layers=int(model_cfg.get("num_layers", 1)), dropout=float(model_cfg.get("dropout", 0.0)),
        embedding_matrix=embedding_matrix, freeze_embeddings=bool(model_cfg.get("freeze_embeddings", False)))
    return JointEmbeddingModel(image_dim, int(model_cfg.get("embedding_dim", 1024)), encoder)


def save_checkpoint(path: str | Path, model: JointEmbeddingModel, *, config: Mapping[str, Any], vocab_size: int,
                    epoch: int, best_metric: Optional[float], history: list, optimizer: Optional[torch.optim.Optimizer] = None,
                    extra: Optional[Mapping[str, Any]] = None) -> None:
    """Save weights + everything needed to rebuild the model (plain tensors / python types only)."""
    payload = {
        "ckpt_version": CKPT_VERSION,
        "model_state": model.state_dict(),
        "config": dict(config),
        "image_dim": model.image_dim,
        "vocab_size": vocab_size,
        "epoch": epoch,
        "best_metric": best_metric,
        "history": history,
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a checkpoint written by :func:`save_checkpoint` (``weights_only`` for safety)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    ckpt = torch.load(p, map_location=map_location, weights_only=True)
    if ckpt.get("ckpt_version") != CKPT_VERSION:
        raise ValueError(f"unsupported checkpoint version {ckpt.get('ckpt_version')!r} (expected {CKPT_VERSION})")
    return ckpt


def model_from_checkpoint(ckpt: Mapping[str, Any]) -> JointEmbeddingModel:
    """Rebuild the model architecture from a loaded checkpoint and restore its weights."""
    cfg = ckpt["config"]["model"]
    model = build_joint_model(cfg, int(ckpt["image_dim"]), int(ckpt["vocab_size"]), embedding_matrix=None)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

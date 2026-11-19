"""Retrieval metrics: Recall@K, mean / median rank (NumPy only).

Directions
----------
* **text -> image** (``t2i``): each caption is a query, all images are candidates, the single image it
  describes is the ground truth. This is the direction the web application implements.
* **image -> text** (``i2t``): each image is a query, all captions are candidates, its 5 captions are the
  ground truth; the rank of the BEST-ranked ground-truth caption is used (standard protocol).

Ties are resolved pessimistically (a candidate with an equal score counts as ranked ahead), so metrics
are never inflated by degenerate, constant scores. Ranks are 1-based.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

DEFAULT_KS = (1, 5, 10)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation (returns float32)."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def ranks_text_to_image(txt_emb: np.ndarray, img_emb: np.ndarray, caption_img_idx: np.ndarray,
                        chunk: int = 512) -> np.ndarray:
    """1-based rank of the ground-truth image for every caption query."""
    n_cap = txt_emb.shape[0]
    ranks = np.empty(n_cap, dtype=np.int64)
    for s in range(0, n_cap, chunk):
        sim = txt_emb[s:s + chunk] @ img_emb.T                      # [c, N_img]
        gt = caption_img_idx[s:s + chunk]
        gt_score = sim[np.arange(sim.shape[0]), gt]
        ranks[s:s + chunk] = (sim >= gt_score[:, None]).sum(axis=1)  # includes the ground truth itself -> 1-based
    return ranks


def ranks_image_to_text(img_emb: np.ndarray, txt_emb: np.ndarray, caption_img_idx: np.ndarray,
                        chunk: int = 256) -> np.ndarray:
    """1-based rank of the best-ranked ground-truth caption for every image query."""
    n_img = img_emb.shape[0]
    ranks = np.empty(n_img, dtype=np.int64)
    for s in range(0, n_img, chunk):
        e = min(n_img, s + chunk)
        sim = img_emb[s:e] @ txt_emb.T                               # [c, N_cap]
        is_gt = caption_img_idx[None, :] == np.arange(s, e)[:, None]
        best = np.where(is_gt, sim, -np.inf).max(axis=1)
        ranks[s:e] = 1 + ((sim >= best[:, None]) & ~is_gt).sum(axis=1)
    return ranks


def recall_at_k(ranks: np.ndarray, ks: Iterable[int] = DEFAULT_KS) -> dict[int, float]:
    """Percentage of queries whose ground truth is ranked within the top K."""
    ranks = np.asarray(ranks)
    return {int(k): float(100.0 * np.mean(ranks <= k)) for k in ks}


def summarize_ranks(ranks: np.ndarray, ks: Iterable[int] = DEFAULT_KS) -> dict[str, float]:
    out = {f"R@{k}": v for k, v in recall_at_k(ranks, ks).items()}
    out["mean_rank"] = float(np.mean(ranks))
    out["median_rank"] = float(np.median(ranks))
    return out


def evaluate_embeddings(img_emb: np.ndarray, txt_emb: np.ndarray, caption_img_idx: np.ndarray,
                        ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """Compute metrics in both directions. Embeddings must already be L2-normalised (cosine similarity).

    Returned keys: ``t2i_R@1``, ``t2i_mean_rank``, ..., ``i2t_R@1`` ..., ``t2i_rsum``, ``i2t_rsum``.
    """
    img_emb = np.asarray(img_emb, dtype=np.float32)
    txt_emb = np.asarray(txt_emb, dtype=np.float32)
    caption_img_idx = np.asarray(caption_img_idx, dtype=np.int64)
    if img_emb.shape[1] != txt_emb.shape[1]:
        raise ValueError(f"embedding dims differ: images {img_emb.shape[1]} vs captions {txt_emb.shape[1]}")
    if txt_emb.shape[0] != caption_img_idx.shape[0]:
        raise ValueError("caption_img_idx must have one entry per caption embedding")
    if caption_img_idx.min() < 0 or caption_img_idx.max() >= img_emb.shape[0]:
        raise ValueError("caption_img_idx contains an out-of-range image index")
    t2i = summarize_ranks(ranks_text_to_image(txt_emb, img_emb, caption_img_idx), ks)
    i2t = summarize_ranks(ranks_image_to_text(img_emb, txt_emb, caption_img_idx), ks)
    out: dict[str, float] = {f"t2i_{k}": v for k, v in t2i.items()}
    out.update({f"i2t_{k}": v for k, v in i2t.items()})
    out["t2i_rsum"] = float(sum(t2i[f"R@{k}"] for k in ks))
    out["i2t_rsum"] = float(sum(i2t[f"R@{k}"] for k in ks))
    return out


def evaluate_protocol(img_emb: np.ndarray, txt_emb: np.ndarray, caption_img_idx: np.ndarray,
                      protocol: str = "full", ks: Sequence[int] = DEFAULT_KS, fold_size: int = 1000) -> dict:
    """Evaluate with the ``full`` candidate set or averaged ``1k`` folds.

    ``1k`` splits the images (in their given order) into consecutive folds of ``fold_size`` images
    (plus their captions) and averages the metrics - the protocol commonly used on the COCO 5k test set.
    Images beyond a full fold are ignored; at least one full fold is required.
    """
    if protocol == "full":
        return {"protocol": "full", "num_images": int(img_emb.shape[0]),
                "num_captions": int(txt_emb.shape[0]), **evaluate_embeddings(img_emb, txt_emb, caption_img_idx, ks)}
    if protocol == "1k":
        n_folds = img_emb.shape[0] // fold_size
        if n_folds < 1:
            raise ValueError(f"protocol '1k' needs at least {fold_size} images, got {img_emb.shape[0]}")
        results = []
        for f in range(n_folds):
            lo, hi = f * fold_size, (f + 1) * fold_size
            cap_mask = (caption_img_idx >= lo) & (caption_img_idx < hi)
            results.append(evaluate_embeddings(img_emb[lo:hi], txt_emb[cap_mask], caption_img_idx[cap_mask] - lo, ks))
        keys = results[0].keys()
        avg = {k: float(np.mean([r[k] for r in results])) for k in keys}
        return {"protocol": "1k", "num_folds": n_folds, "fold_size": fold_size,
                "num_images": n_folds * fold_size, **avg}
    raise ValueError(f"unknown protocol '{protocol}' (use 'full' or '1k')")


def higher_is_better(metric: str) -> bool:
    return not (metric.endswith("mean_rank") or metric.endswith("median_rank"))


def format_metrics(metrics: Mapping[str, float]) -> str:
    keys = [k for k in metrics if k.startswith(("t2i_", "i2t_")) and not k.endswith("rsum")]
    return "  ".join(f"{k}={metrics[k]:.2f}" for k in keys)

"""Bidirectional hinge ranking loss for image-text embeddings.

For a batch of ``B`` matched pairs with unit-norm embeddings ``i_k`` (image) and ``t_k`` (text),
let ``s(i, t) = i . t`` (cosine similarity) and ``alpha`` the margin. The loss is

    L = 1/B * sum_k [ sum_{j != k} max(0, alpha - s(i_k, t_k) + s(i_k, t_j))     (wrong captions for image k)
                    + sum_{j != k} max(0, alpha - s(i_k, t_k) + s(i_j, t_k)) ]   (wrong images for caption k)

i.e. every matching pair must score at least ``alpha`` higher than every contrasting pair in the batch.
With ``max_violation=True`` only the hardest negative per query is used (VSE++ variant).

Several captions of the same image can land in one batch; such pairs are NOT negatives, so they are
masked out when ``image_index`` is supplied. The torch function and the loop-based NumPy reference
below are tested against each other.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def ranking_loss_numpy(img: np.ndarray, txt: np.ndarray, margin: float = 0.2,
                       image_index: Optional[np.ndarray] = None, max_violation: bool = False) -> float:
    """Slow, explicit reference implementation (used for testing)."""
    b = img.shape[0]
    idx = np.arange(b) if image_index is None else np.asarray(image_index)
    scores = img @ txt.T
    total = 0.0
    for k in range(b):
        neg_c = [max(0.0, margin - scores[k, k] + scores[k, j]) for j in range(b) if idx[j] != idx[k]]
        neg_i = [max(0.0, margin - scores[k, k] + scores[j, k]) for j in range(b) if idx[j] != idx[k]]
        if max_violation:
            total += (max(neg_c) if neg_c else 0.0) + (max(neg_i) if neg_i else 0.0)
        else:
            total += sum(neg_c) + sum(neg_i)
    return float(total / b)


def ranking_loss(img, txt, margin: float = 0.2, image_index=None, max_violation: bool = False):
    """Vectorised PyTorch implementation. ``img`` and ``txt`` are L2-normalised [B, D] tensors."""
    import torch

    b = img.shape[0]
    scores = img @ txt.t()                                   # scores[i, j] = s(image_i, text_j)
    diag = scores.diag().view(-1, 1)                         # s(image_k, text_k) of the matching pairs
    cost_txt = (margin + scores - diag).clamp(min=0)         # row k: wrong captions for image k
    cost_img = (margin + scores - diag.t()).clamp(min=0)     # column k: wrong images for caption k
    if image_index is None:
        pos = torch.eye(b, dtype=torch.bool, device=scores.device)
    else:
        image_index = image_index.to(scores.device)
        pos = image_index.view(-1, 1) == image_index.view(1, -1)  # includes the diagonal
    cost_txt = cost_txt.masked_fill(pos, 0.0)
    cost_img = cost_img.masked_fill(pos, 0.0)
    if max_violation:
        return (cost_txt.max(dim=1)[0].sum() + cost_img.max(dim=0)[0].sum()) / b
    return (cost_txt.sum() + cost_img.sum()) / b

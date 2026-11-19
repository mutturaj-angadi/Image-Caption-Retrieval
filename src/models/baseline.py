"""Multi-response linear regression baseline (NumPy only).

Caption representation  x  = mean of the word vectors of the caption's tokens (GloVe, 300-d).
Image representation    y  = VGG-19 fc7 feature (4,096-d).
Model:                  y_hat = [x, 1] W    with W the ridge-regularised least-squares solution

    W = (X'X + lambda * R)^(-1) X'Y        (R = identity, but the bias row is not penalised)

Retrieval ranks images by cosine similarity between ``y_hat`` and each image's fc7 feature. Ridge
regularisation (lambda) is an engineering addition for numerical stability; lambda = 0 recovers plain
multi-response least squares.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class LinearRegressionBaseline:
    def __init__(self, l2: float = 1.0):
        self.l2 = float(l2)
        self.W: Optional[np.ndarray] = None       # [D + 1, F]
        self._xtx: Optional[np.ndarray] = None
        self._xty: Optional[np.ndarray] = None
        self.num_samples = 0

    # -- fitting -----------------------------------------------------------
    def accumulate(self, X: np.ndarray, Y: np.ndarray, image_index: Optional[np.ndarray] = None,
                   chunk: int = 8192) -> "LinearRegressionBaseline":
        """Accumulate the normal equations. ``Y[image_index[i]]`` is the target of caption ``i``.

        Chunking keeps memory bounded: the (N_captions x 4096) target matrix is never materialised.
        """
        if image_index is None:
            if X.shape[0] != Y.shape[0]:
                raise ValueError("X and Y must have the same number of rows when image_index is not given")
            image_index = np.arange(X.shape[0])
        n, d = X.shape
        f = Y.shape[1]
        self._xtx = np.zeros((d + 1, d + 1), dtype=np.float64)
        self._xty = np.zeros((d + 1, f), dtype=np.float64)
        for s in range(0, n, chunk):
            xb = np.hstack([X[s:s + chunk].astype(np.float64), np.ones((min(chunk, n - s), 1))])
            yb = np.asarray(Y[image_index[s:s + chunk]], dtype=np.float64)
            self._xtx += xb.T @ xb
            self._xty += xb.T @ yb
        self.num_samples = n
        return self

    def solve(self, l2: Optional[float] = None) -> "LinearRegressionBaseline":
        """Solve for ``W`` with ridge strength ``l2`` using the accumulated statistics."""
        if self._xtx is None or self._xty is None:
            raise RuntimeError("call accumulate() before solve()")
        if l2 is not None:
            self.l2 = float(l2)
        reg = self.l2 * np.eye(self._xtx.shape[0])
        reg[-1, -1] = 0.0  # do not penalise the intercept
        self.W = np.linalg.solve(self._xtx + reg, self._xty).astype(np.float32)
        return self

    def fit(self, X: np.ndarray, Y: np.ndarray, image_index: Optional[np.ndarray] = None) -> "LinearRegressionBaseline":
        return self.accumulate(X, Y, image_index).solve()

    # -- inference ---------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] + 1 != self.W.shape[0]:
            raise ValueError(f"expected {self.W.shape[0] - 1}-d inputs, got {X.shape[1]}")
        return X @ self.W[:-1] + self.W[-1]

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path, word_vectors: Optional[np.ndarray] = None) -> None:
        if self.W is None:
            raise RuntimeError("model is not fitted")
        payload = {"W": self.W, "l2": np.asarray(self.l2), "format": np.asarray("ncir-baseline-v1")}
        if word_vectors is not None:
            payload["word_vectors"] = np.asarray(word_vectors, dtype=np.float32)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> tuple["LinearRegressionBaseline", Optional[np.ndarray]]:
        with np.load(path, allow_pickle=False) as z:
            if str(z["format"]) != "ncir-baseline-v1":
                raise ValueError(f"{Path(path).name} is not an ncir-baseline-v1 file")
            model = cls(float(z["l2"]))
            model.W = z["W"].astype(np.float32)
            wv = z["word_vectors"].astype(np.float32) if "word_vectors" in z.files else None
        return model, wv

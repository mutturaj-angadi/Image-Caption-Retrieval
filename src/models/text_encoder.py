"""Recurrent caption encoder (GRU or LSTM) initialised from pretrained word vectors. Requires PyTorch."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextEncoder(nn.Module):
    """tokens -> embedding layer (GloVe init) -> GRU/LSTM -> last hidden state -> linear -> L2-normalise.

    Inputs: ``ids`` int64 [B, L] (0 = padding) and ``lengths`` int64 [B] with ``lengths >= 1``.
    Output: unit-norm tensor [B, embedding_dim].
    """

    def __init__(self, vocab_size: int, word_dim: int, hidden_dim: int, embedding_dim: int, rnn_type: str = "gru",
                 num_layers: int = 1, dropout: float = 0.0, embedding_matrix: Optional[np.ndarray] = None,
                 freeze_embeddings: bool = False):
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in ("gru", "lstm"):
            raise ValueError(f"rnn_type must be 'gru' or 'lstm', got '{rnn_type}'")
        self.rnn_type = rnn_type
        self.embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        if embedding_matrix is not None:
            if embedding_matrix.shape != (vocab_size, word_dim):
                raise ValueError(f"embedding matrix shape {embedding_matrix.shape} != {(vocab_size, word_dim)}")
            with torch.no_grad():
                self.embed.weight.copy_(torch.as_tensor(embedding_matrix))
                self.embed.weight[0].zero_()
        self.embed.weight.requires_grad = not freeze_embeddings
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(word_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embed(ids))
        packed = pack_padded_sequence(x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):  # LSTM returns (h_n, c_n)
            hidden = hidden[0]
        last = hidden[-1]              # final hidden state of the top layer: [B, hidden_dim]
        return F.normalize(self.proj(self.dropout(last)), dim=-1)

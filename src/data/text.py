"""Caption cleaning, tokenisation, vocabulary, padding and word-vector loading."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.utils.common import load_json, save_json

PAD = "<pad>"
UNK = "<unk>"
PAD_ID = 0
UNK_ID = 1

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SPACE_RE = re.compile(r"\s+")


def clean_caption(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return _SPACE_RE.sub(" ", str(text).strip().lower())


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenisation (punctuation is dropped, "don't" -> "don", "t")."""
    return _TOKEN_RE.findall(clean_caption(text))


class Vocabulary:
    """Token <-> id mapping with fixed ``<pad>`` = 0 and ``<unk>`` = 1."""

    def __init__(self, itos: Sequence[str]):
        itos = list(itos)
        if len(itos) < 2 or itos[PAD_ID] != PAD or itos[UNK_ID] != UNK:
            raise ValueError("Vocabulary must start with '<pad>', '<unk>'")
        if len(set(itos)) != len(itos):
            raise ValueError("Vocabulary contains duplicate tokens")
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, token_lists: Iterable[Sequence[str]], min_freq: int = 1, max_size: int | None = None) -> "Vocabulary":
        """Build from tokenised captions (use TRAIN captions only to avoid leakage)."""
        counts: Counter[str] = Counter()
        for toks in token_lists:
            counts.update(toks)
        words = [w for w, c in counts.items() if c >= min_freq and w not in (PAD, UNK)]
        words.sort(key=lambda w: (-counts[w], w))  # deterministic order
        if max_size is not None:
            words = words[: max(0, max_size - 2)]
        return cls([PAD, UNK] + words)

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, UNK_ID)

    def unknown_tokens(self, tokens: Iterable[str]) -> list[str]:
        return [t for t in tokens if t not in self.stoi]

    def encode(self, tokens: Sequence[str], max_len: int | None = None) -> list[int]:
        ids = [self.token_to_id(t) for t in tokens]
        if max_len is not None:
            ids = ids[:max_len]
        return ids or [UNK_ID]  # never return an empty sequence (RNN packing needs length >= 1)

    def encode_batch(self, captions: Sequence[str], max_len: int = 32, pad_to: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Encode captions to ``ids`` (int32, [B, L]) and ``lengths`` (int64, [B]).

        ``L`` is ``pad_to`` if given, else the longest sequence in the batch. Padding id is 0.
        """
        encoded = [self.encode(tokenize(c), max_len) for c in captions]
        lengths = np.asarray([len(e) for e in encoded], dtype=np.int64)
        width = pad_to if pad_to is not None else int(lengths.max(initial=1))
        ids = np.zeros((len(encoded), width), dtype=np.int32)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = e[:width]
        return ids, np.minimum(lengths, width)

    def save(self, path: str | Path) -> None:
        save_json({"format": "ncir-vocab-v1", "itos": self.itos}, path)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        data = load_json(path)
        return cls(data["itos"])


# ---------------------------------------------------------------- word vectors
def load_glove_matrix(path: str | Path, vocab: Vocabulary, dim: int = 300, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Build an embedding matrix [V, dim] from a GloVe text file.

    Only words that are in ``vocab`` are parsed. Words missing from GloVe get small random
    vectors (seeded); ``<pad>`` is all zeros. Returns ``(matrix, stats)``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"GloVe file not found: {p}. Download glove.6B.zip with scripts/download_data.py --what glove "
            "(or set data.glove_path to null for synthetic smoke-test vectors)."
        )
    wanted = set(vocab.itos[2:])
    found: dict[str, np.ndarray] = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            word, _, rest = line.partition(" ")
            if word in wanted and word not in found:
                vec = np.asarray(rest.split(), dtype=np.float32)
                if vec.shape[0] != dim:
                    raise ValueError(f"Expected {dim}-d GloVe vectors but '{word}' has {vec.shape[0]} values")
                found[word] = vec
                if len(found) == len(wanted):
                    break
    rng = np.random.RandomState(seed)
    matrix = rng.normal(0.0, 0.3, size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD_ID] = 0.0
    for word, vec in found.items():
        matrix[vocab.stoi[word]] = vec
    n_words = max(1, len(wanted))
    stats = {
        "source": f"glove:{p.name}",
        "dim": dim,
        "vocab_size": len(vocab),
        "num_found": len(found),
        "num_missing": len(wanted) - len(found),
        "coverage": len(found) / n_words,
    }
    return matrix, stats


def synthetic_word_vectors(vocab: Vocabulary, dim: int = 300, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Deterministic pseudo word vectors derived from a hash of each word.

    FOR SMOKE TESTS ONLY - these carry no semantic information and are not GloVe.
    """
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    for tok, i in vocab.stoi.items():
        if i == PAD_ID:
            continue
        digest = hashlib.sha256(f"{seed}:{tok}".encode("utf-8")).digest()
        rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
        matrix[i] = rng.normal(0.0, 0.3, size=dim).astype(np.float32)
    return matrix, {"source": "synthetic-hash (NOT GloVe; smoke tests only)", "dim": dim,
                    "vocab_size": len(vocab), "coverage": 1.0}


def random_word_vectors(vocab: Vocabulary, dim: int = 300, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Random-initialised embedding matrix (used for the 'no pretrained GloVe' ablation)."""
    rng = np.random.RandomState(seed)
    matrix = rng.normal(0.0, 0.3, size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD_ID] = 0.0
    return matrix, {"source": "random-init", "dim": dim, "vocab_size": len(vocab), "coverage": 0.0}


def mean_word_vectors(captions: Sequence[str], vocab: Vocabulary, word_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average the word vectors of the in-vocabulary tokens of each caption.

    Returns ``(features [N, dim] float32, known_counts [N] int)``. Captions with no known token
    get a zero vector (``known_counts == 0``).
    """
    feats = np.zeros((len(captions), word_vectors.shape[1]), dtype=np.float32)
    known = np.zeros(len(captions), dtype=np.int64)
    for i, cap in enumerate(captions):
        ids = [vocab.stoi[t] for t in tokenize(cap) if t in vocab.stoi]
        known[i] = len(ids)
        if ids:
            feats[i] = word_vectors[ids].mean(axis=0)
    return feats, known

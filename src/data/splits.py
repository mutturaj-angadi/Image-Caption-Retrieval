"""Reproducible subsetting and split helpers."""
from __future__ import annotations

import random
from typing import Mapping, Optional

from src.data.records import ImageRecord, VALID_SPLITS


def group_by_split(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    out: dict[str, list[ImageRecord]] = {s: [] for s in VALID_SPLITS}
    for r in records:
        out[r.split].append(r)
    return out


def subset_records(records: list[ImageRecord], max_per_split: Optional[Mapping[str, Optional[int]]],
                   seed: int = 42) -> list[ImageRecord]:
    """Deterministically keep at most ``max_per_split[split]`` images per split.

    Sampling is done on image ids sorted lexicographically and shuffled with ``seed``, so the
    same arguments always return the same subset. Splits are never mixed, so no leakage is
    introduced. ``None`` / missing limits keep the whole split.
    """
    if not max_per_split:
        return list(records)
    out: list[ImageRecord] = []
    for split, recs in group_by_split(records).items():
        recs = sorted(recs, key=lambda r: r.image_id)
        limit = max_per_split.get(split)
        if limit is not None and limit < len(recs):
            idx = list(range(len(recs)))
            random.Random(f"{seed}:{split}").shuffle(idx)
            keep = sorted(idx[:limit])
            recs = [recs[i] for i in keep]
        out.extend(recs)
    return out

"""Aligned image-caption arrays used by training and evaluation (NumPy only)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.records import ImageRecord
from src.data.text import Vocabulary
from src.utils.common import load_json


@dataclass
class SplitData:
    """Everything needed to train / evaluate on one split.

    ``features[i]`` belongs to ``image_ids[i]``; ``captions[j]`` describes image
    ``caption_img_idx[j]`` (an index into ``image_ids``).
    """

    split: str
    image_ids: list[str]
    features: np.ndarray            # [N_img, F] float32
    captions: list[str]             # [N_cap]
    caption_img_idx: np.ndarray     # [N_cap] int64
    cap_ids: np.ndarray             # [N_cap, L] int32 (padded with 0)
    cap_lengths: np.ndarray         # [N_cap] int64

    @property
    def num_images(self) -> int:
        return len(self.image_ids)

    @property
    def num_captions(self) -> int:
        return len(self.captions)


def build_split_data(split: str, records: Sequence[ImageRecord], image_ids: Sequence[str], features: np.ndarray,
                     vocab: Vocabulary, max_len: int) -> SplitData:
    """Align records with a feature matrix and encode all captions.

    ``image_ids`` is the row order of ``features``. Every id must have a record in ``split``.
    """
    by_id = {r.image_id: r for r in records if r.split == split}
    missing = [i for i in image_ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} feature rows have no '{split}' record (e.g. {missing[:3]}). "
                         "Re-run extract_features.py after prepare_data.py.")
    captions: list[str] = []
    idx: list[int] = []
    for row, image_id in enumerate(image_ids):
        for cap in by_id[image_id].captions:
            captions.append(cap)
            idx.append(row)
    ids, lengths = vocab.encode_batch(captions, max_len=max_len, pad_to=max_len)
    return SplitData(split, list(image_ids), features, captions, np.asarray(idx, dtype=np.int64), ids, lengths)


def load_records(metadata_path: str | Path) -> tuple[list[ImageRecord], dict]:
    """Load ``metadata.json`` written by ``prepare_data.py``; returns ``(records, dataset_info)``."""
    data = load_json(metadata_path)
    if data.get("format") != "ncir-metadata-v1":
        raise ValueError(f"{metadata_path} is not an ncir-metadata-v1 file")
    return [ImageRecord.from_dict(d) for d in data["records"]], data.get("dataset", {}) | {
        "images_root": data.get("images_root")}

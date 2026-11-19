"""Dataset records and loaders for COCO-style caption annotations.

Every loader produces a list of :class:`ImageRecord`. ``file_name`` is always a POSIX path
*relative to the images root*, so processed metadata never contains machine-specific paths.
"""
from __future__ import annotations

import posixpath
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from src.utils.common import load_json

VALID_SPLITS = ("train", "val", "test")


@dataclass
class ImageRecord:
    image_id: str
    file_name: str
    captions: list[str] = field(default_factory=list)
    split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImageRecord":
        return cls(
            image_id=str(d["image_id"]),
            file_name=str(d["file_name"]),
            captions=[str(c) for c in d.get("captions", [])],
            split=str(d.get("split", "train")),
        )


def load_karpathy(json_path: str | Path, restval_to_train: bool = True) -> list[ImageRecord]:
    """Load the Karpathy & Fei-Fei COCO split file (``dataset_coco.json``).

    The historical 113,287 / 5,000 / 5,000 split used by the paper corresponds to this file when
    the ``restval`` images (30,504) are merged into training (82,783 + 30,504 = 113,287).
    """
    data = load_json(json_path)
    if "images" not in data:
        raise ValueError(f"{json_path} does not look like a Karpathy split file (no 'images' key)")
    records: list[ImageRecord] = []
    for item in data["images"]:
        split = item.get("split", "train")
        if split == "restval":
            split = "train" if restval_to_train else "val"
        if split not in VALID_SPLITS:
            continue
        captions = [s.get("raw", " ".join(s.get("tokens", []))) for s in item.get("sentences", [])]
        filepath = item.get("filepath", "")
        rel = posixpath.join(filepath, item["filename"]) if filepath else item["filename"]
        image_id = str(item.get("cocoid", posixpath.splitext(item["filename"])[0]))
        records.append(ImageRecord(image_id=image_id, file_name=rel, captions=captions, split=split))
    return records


def load_coco_captions(json_path: str | Path, image_subdir: str = "") -> list[ImageRecord]:
    """Load an official COCO ``captions_*.json`` file (images + annotations).

    All records get split ``"train"``; use :func:`make_random_split` to assign real splits.
    """
    data = load_json(json_path)
    if "images" not in data or "annotations" not in data:
        raise ValueError(f"{json_path} is not a COCO captions file (needs 'images' and 'annotations')")
    caps: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        caps.setdefault(int(ann["image_id"]), []).append(str(ann["caption"]).strip())
    records = []
    for img in data["images"]:
        rel = posixpath.join(image_subdir, img["file_name"]) if image_subdir else img["file_name"]
        records.append(ImageRecord(image_id=str(img["id"]), file_name=rel,
                                   captions=caps.get(int(img["id"]), []), split="train"))
    return records


def make_random_split(records: Iterable[ImageRecord], n_val: int, n_test: int, seed: int = 42) -> list[ImageRecord]:
    """Deterministically assign splits: ``n_val`` / ``n_test`` random images, the rest train.

    This is *Karpathy-style* but NOT the identical image assignment of the historical split;
    document it whenever it is used.
    """
    recs = sorted(records, key=lambda r: r.image_id)
    if n_val + n_test >= len(recs):
        raise ValueError("n_val + n_test must be smaller than the number of images")
    order = list(range(len(recs)))
    random.Random(seed).shuffle(order)
    val_idx = set(order[:n_val])
    test_idx = set(order[n_val:n_val + n_test])
    out = []
    for i, r in enumerate(recs):
        split = "val" if i in val_idx else "test" if i in test_idx else "train"
        out.append(ImageRecord(r.image_id, r.file_name, list(r.captions), split))
    return out


def standardize_captions(records: Iterable[ImageRecord], per_image: Optional[int] = 5) -> tuple[list[ImageRecord], dict]:
    """Strip captions, drop empty ones, keep the first ``per_image`` and drop images with fewer.

    Using exactly five captions per image keeps the evaluation protocol identical to the paper's.
    """
    kept: list[ImageRecord] = []
    dropped_no_caps: list[str] = []
    empty_removed = 0
    for r in records:
        caps = [c.strip() for c in r.captions]
        empty_removed += sum(1 for c in caps if not c)
        caps = [c for c in caps if c]
        if per_image is not None:
            if len(caps) < per_image:
                dropped_no_caps.append(r.image_id)
                continue
            caps = caps[:per_image]
        elif not caps:
            dropped_no_caps.append(r.image_id)
            continue
        kept.append(ImageRecord(r.image_id, r.file_name, caps, r.split))
    return kept, {
        "empty_captions_removed": empty_removed,
        "images_dropped_too_few_captions": len(dropped_no_caps),
        "dropped_image_ids_sample": dropped_no_caps[:20],
        "captions_per_image": per_image,
    }

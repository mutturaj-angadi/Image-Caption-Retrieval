"""Tiny SYNTHETIC image-caption dataset for smoke tests and automated tests.

The images are procedurally drawn shapes and the captions are templates over the same attributes.
This data exists only to exercise the pipeline without downloading COCO. It must never be reported
as an experimental result or described as COCO.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.data.records import ImageRecord

COLORS = {"red": (220, 40, 40), "green": (40, 180, 70), "blue": (50, 80, 220),
          "yellow": (235, 210, 40), "purple": (150, 60, 190)}
BACKGROUNDS = {"white": (245, 245, 245), "black": (20, 20, 20)}
SHAPES = ("circle", "square", "triangle")
SIZES = {"small": 12, "large": 24}
POSITIONS = {"center": (32, 32), "left": (18, 32), "right": (46, 32), "top": (32, 18), "bottom": (32, 46)}

_TEMPLATES = [
    "a {color} {shape} on a {bg} background",
    "a {size} {color} {shape}",
    "a {color} {shape} on the {pos} of a {bg} image",
    "{bg} background with a {size} {color} {shape}",
    "a {size} {color} {shape} at the {pos}",
    "a picture of a {color} {shape}",
]


def _draw(attrs: dict, rng: random.Random, size: int = 64) -> Image.Image:
    bg = tuple(max(0, min(255, c + rng.randint(-8, 8))) for c in BACKGROUNDS[attrs["bg"]])
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    cx, cy = POSITIONS[attrs["pos"]]
    cx += rng.randint(-2, 2)
    cy += rng.randint(-2, 2)
    r = SIZES[attrs["size"]] // 2 + rng.randint(-1, 1)
    col = tuple(max(0, min(255, c + rng.randint(-12, 12))) for c in COLORS[attrs["color"]])
    if attrs["shape"] == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif attrs["shape"] == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=col)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=col)
    arr = np.asarray(im, dtype=np.int16) + rng.randint(-3, 3)  # global brightness jitter
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_synthetic_dataset(root: str | Path, n_train: int = 150, n_val: int = 30, n_test: int = 30,
                           seed: int = 7) -> tuple[list[ImageRecord], Path]:
    """Write synthetic images under ``root/images`` and return ``(records, images_root)``."""
    rng = random.Random(seed)
    images_root = Path(root) / "images"
    combos = [dict(color=c, shape=s, bg=b, size=z, pos=p) for c in COLORS for s in SHAPES
              for b in BACKGROUNDS for z in SIZES for p in POSITIONS]
    rng.shuffle(combos)
    need = n_train + n_val + n_test
    while len(combos) < need:
        combos.extend(dict(c) for c in combos[: need - len(combos)])
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    records: list[ImageRecord] = []
    for i, (attrs, split) in enumerate(zip(combos[:need], splits)):
        image_id = f"syn_{i:05d}"
        rel = f"{split}/{image_id}.png"
        path = images_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        _draw(attrs, rng).save(path)
        templates = rng.sample(_TEMPLATES, 5)
        caps = [t.format(color=attrs["color"], shape=attrs["shape"], bg=attrs["bg"], size=attrs["size"],
                         pos=attrs["pos"]) for t in templates]
        records.append(ImageRecord(image_id=image_id, file_name=rel, captions=caps, split=split))
    (Path(root) / "README.txt").write_text(
        "SYNTHETIC smoke-test data (procedurally generated shapes). Not COCO. Never report metrics from this "
        "data as experimental results.\n", encoding="utf-8")
    return records, images_root

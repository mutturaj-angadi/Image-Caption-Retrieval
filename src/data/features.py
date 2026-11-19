"""Image feature extraction (VGG-19 fc7) with on-disk caching.

Two extractors are provided:

* ``vgg19_fc7``  - the research-inspired 4,096-d fc7 feature of a pretrained VGG-19 (torchvision).
* ``simple_rgb256`` - a tiny NumPy colour/layout descriptor used ONLY by the smoke tests so the
  pipeline can run without PyTorch or a dataset download. It is not a research feature.

Differences from the paper (documented, not hidden): torchvision's ImageNet weights are used instead of the
original Oxford/Caffe VGG-19 model, with a single 224x224 centre crop after resizing to 256.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
from PIL import Image

from src.data.records import ImageRecord
from src.utils.common import get_logger, load_json, progress, save_json

log = get_logger("ncir.features")


class FeatureExtractionError(RuntimeError):
    """Raised when an image cannot be read or a feature tensor is invalid."""


class FeatureExtractor(Protocol):
    name: str
    dim: int

    def extract(self, paths: Sequence[Path]) -> np.ndarray: ...


def _load_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise FeatureExtractionError(f"cannot read image {path}: {exc}") from exc


class SimpleImageFeatures:
    """8x8 average-pooled RGB (192) + 4x4x4 colour histogram (64) = 256 dims. Smoke tests only."""

    name = "simple_rgb256"
    dim = 256

    def extract(self, paths: Sequence[Path]) -> np.ndarray:
        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        for i, p in enumerate(paths):
            im = _load_rgb(Path(p))
            small = np.asarray(im.resize((8, 8), Image.BILINEAR), dtype=np.float32) / 255.0
            q = (np.asarray(im.resize((32, 32), Image.BILINEAR)) // 64).reshape(-1, 3)  # 0..3 per channel
            hist = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64).astype(np.float32)
            hist /= max(1.0, hist.sum())
            out[i] = np.concatenate([small.reshape(-1), hist])
        return out


class VGG19Fc7Extractor:
    """Pretrained VGG-19; returns the 4,096-d activation after the second FC layer's ReLU (fc7)."""

    name = "vgg19_fc7"
    dim = 4096

    def __init__(self, device: str = "auto", batch_size: int = 32, num_workers: int = 0):
        try:
            import torch
            import torchvision
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError("vgg19_fc7 requires torch and torchvision (see requirements.txt)") from exc
        self._torch = torch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else
                                   ("cpu" if device == "auto" else device))
        weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1  # downloads ~550 MB on first use
        model = torchvision.models.vgg19(weights=weights)
        model.eval()
        self.features = model.features.to(self.device)
        self.avgpool = model.avgpool.to(self.device)
        # classifier = [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear]; fc7 is output of index 4.
        self.fc = model.classifier[:5].to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, paths: Sequence[Path]) -> np.ndarray:
        torch = self._torch
        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                chunk = paths[start:start + self.batch_size]
                batch = torch.stack([self.transform(_load_rgb(Path(p))) for p in chunk]).to(self.device)
                x = self.avgpool(self.features(batch))
                x = self.fc(torch.flatten(x, 1))
                out[start:start + len(chunk)] = x.float().cpu().numpy()
        return out


def get_extractor(name: str, **kwargs) -> FeatureExtractor:
    if name == "vgg19_fc7":
        return VGG19Fc7Extractor(**kwargs)
    if name == "simple_rgb256":
        return SimpleImageFeatures()
    raise ValueError(f"unknown feature extractor '{name}' (choose vgg19_fc7 or simple_rgb256)")


# ------------------------------------------------------------------ caching
def _paths(features_dir: Path, split: str) -> tuple[Path, Path]:
    return features_dir / f"{split}_features.npy", features_dir / f"{split}_ids.json"


def check_features(feats: np.ndarray, expected_n: int, expected_dim: int) -> None:
    """Verify shape and numerical validity of a feature matrix."""
    if feats.shape != (expected_n, expected_dim):
        raise FeatureExtractionError(f"feature shape {feats.shape} != expected {(expected_n, expected_dim)}")
    if not np.isfinite(feats).all():
        raise FeatureExtractionError("features contain NaN or Inf")


def extract_split_features(records: Sequence[ImageRecord], images_root: str | Path, features_dir: str | Path,
                           extractor: FeatureExtractor, chunk: int = 256, force: bool = False) -> dict:
    """Extract and cache features for every split present in ``records``.

    Writes ``<split>_features.npy`` (float32, rows aligned with ``<split>_ids.json``) and
    ``extractor.json``. A split is skipped when a cache with identical ids and extractor exists.
    """
    root = Path(images_root)
    fdir = Path(features_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"extractor": extractor.name, "dim": extractor.dim, "splits": {}}
    for split in ("train", "val", "test"):
        recs = sorted((r for r in records if r.split == split), key=lambda r: r.image_id)
        if not recs:
            continue
        ids = [r.image_id for r in recs]
        fpath, ipath = _paths(fdir, split)
        if not force and fpath.is_file() and ipath.is_file():
            meta_p = fdir / "extractor.json"
            cached_ok = meta_p.is_file() and load_json(meta_p).get("name") == extractor.name
            if cached_ok and load_json(ipath) == ids:
                try:
                    arr = np.load(fpath, mmap_mode="r")
                    check_features(arr, len(ids), extractor.dim)
                    log.info("features for split '%s' found in cache (%d images) - skipping", split, len(ids))
                    summary["splits"][split] = {"n": len(ids), "cached": True}
                    continue
                except (FeatureExtractionError, ValueError, OSError):
                    log.warning("cache for '%s' is invalid; recomputing", split)
        tmp = fpath.with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(ids), extractor.dim))
        for start in progress(range(0, len(recs), chunk), total=(len(recs) + chunk - 1) // chunk,
                              desc=f"features[{split}]"):
            part = recs[start:start + chunk]
            out[start:start + len(part)] = extractor.extract([root / r.file_name for r in part])
        out.flush()
        check_features(np.asarray(out), len(ids), extractor.dim)
        del out
        os.replace(tmp, fpath)
        save_json(ids, ipath)
        summary["splits"][split] = {"n": len(ids), "cached": False}
        log.info("extracted %d features for split '%s'", len(ids), split)
    save_json({"name": extractor.name, "dim": extractor.dim}, fdir / "extractor.json")
    return summary


def load_features(features_dir: str | Path, split: str, mmap: bool = False) -> tuple[np.ndarray, list[str]]:
    """Load ``(features, image_ids)`` for ``split``; rows of ``features`` match ``image_ids``."""
    fpath, ipath = _paths(Path(features_dir), split)
    if not fpath.is_file() or not ipath.is_file():
        raise FileNotFoundError(
            f"Features for split '{split}' not found in {features_dir}. Run scripts/extract_features.py first."
        )
    feats = np.load(fpath, mmap_mode="r" if mmap else None)
    ids = [str(i) for i in load_json(ipath)]
    if feats.shape[0] != len(ids):
        raise FeatureExtractionError(f"{fpath.name} has {feats.shape[0]} rows but {len(ids)} ids")
    return feats, ids

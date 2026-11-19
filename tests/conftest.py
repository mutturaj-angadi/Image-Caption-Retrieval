"""Shared fixtures. All data used by the tests is SYNTHETIC (see src/data/synthetic.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.smoke import build_smoke_workspace  # noqa: E402


@pytest.fixture(scope="session")
def smoke_ws(tmp_path_factory):
    """A complete NumPy-only smoke workspace (data, features, baseline model, index)."""
    return build_smoke_workspace(tmp_path_factory.mktemp("smoke"), with_neural=False)


@pytest.fixture()
def tiny_images(tmp_path):
    """Three small valid PNG files plus helper records."""
    from PIL import Image

    from src.data.records import ImageRecord

    root = tmp_path / "imgs"
    root.mkdir()
    records = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        Image.new("RGB", (16, 16), color).save(root / f"im{i}.png")
        records.append(ImageRecord(f"id{i}", f"im{i}.png", [f"caption number {i} a", f"caption number {i} b"],
                                   ["train", "val", "test"][i]))
    return root, records

"""Download COCO 2014 images/annotations, the Karpathy split file and GloVe 6B vectors.

    python scripts/download_data.py --what karpathy glove annotations     # small (~1.2 GB)
    python scripts/download_data.py --what images                         # large (~19 GB)

Files are streamed to ``data/raw`` and skipped when already present. If your network blocks a
download, fetch the URL manually in a browser and place the file as described in data/README.md.
Check the licence/terms in NOTICE.md before use.
"""
import _bootstrap  # noqa: F401  (must be first)

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from src.utils.common import resolve_path

SOURCES = {
    "annotations": ("http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                    "coco/annotations_trainval2014.zip", "coco"),
    "train_images": ("http://images.cocodataset.org/zips/train2014.zip", "coco/train2014.zip", "coco"),
    "val_images": ("http://images.cocodataset.org/zips/val2014.zip", "coco/val2014.zip", "coco"),
    "karpathy": ("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
                 "karpathy/caption_datasets.zip", "karpathy"),
    "glove": ("https://nlp.stanford.edu/data/glove.6B.zip", "glove/glove.6B.zip", "glove"),
}
GROUPS = {"images": ["train_images", "val_images"], "all": list(SOURCES)}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f}%)", end="", flush=True)
    print()
    tmp.replace(dest)


def extract(zip_path: Path, target: Path, only: str | None = None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if only and not member.endswith(only):
                continue
            dest = (target / member).resolve()
            if target.resolve() not in dest.parents and dest != target.resolve():
                raise RuntimeError(f"unsafe path in archive: {member}")  # zip-slip protection
            z.extract(member, target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", nargs="+", default=["karpathy", "glove", "annotations"],
                    choices=sorted(SOURCES) + sorted(GROUPS), help="what to download")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--keep-zips", action="store_true", help="do not delete archives after extraction")
    args = ap.parse_args()
    raw = resolve_path(args.raw_dir)
    wanted: list[str] = []
    for w in args.what:
        wanted += GROUPS.get(w, [w])
    free_gb = shutil.disk_usage(raw.parent if raw.exists() else raw.parent.parent).free / 1e9
    print(f"free disk space: {free_gb:.1f} GB")
    for key in dict.fromkeys(wanted):
        url, rel, sub = SOURCES[key]
        zpath = raw / rel
        print(f"[{key}] {url}")
        try:
            if not zpath.exists():
                download(url, zpath)
            only = "glove.6B.300d.txt" if key == "glove" else None
            extract(zpath, raw / sub, only)
            if not args.keep_zips:
                zpath.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}\n  -> download it manually and place it as described in data/README.md",
                  file=sys.stderr)
            return 1
    print("done. Next: python scripts/prepare_data.py --source karpathy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

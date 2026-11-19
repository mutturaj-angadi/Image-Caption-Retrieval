"""Extract and cache image features for every split in data/processed/metadata.json.

    python scripts/extract_features.py                     # VGG-19 fc7 (4096-d); GPU is used automatically if present
    python scripts/extract_features.py --batch-size 8      # lower this if you run out of memory / VRAM
    python scripts/extract_features.py --extractor simple_rgb256   # smoke tests only

To limit the amount of work, create a subset with prepare_data.py (--max-train ...) first.
Cached features are reused automatically; pass --force to recompute.
"""
import _bootstrap  # noqa: F401

import argparse
import sys

from src.data.features import extract_split_features, get_extractor
from src.data.pairs import load_records
from src.utils.common import environment_info, resolve_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out", default=None, help="default: <processed-dir>/features")
    ap.add_argument("--extractor", default="vgg19_fc7", choices=["vgg19_fc7", "simple_rgb256"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    processed = resolve_path(args.processed_dir)
    try:
        records, info = load_records(processed / "metadata.json")
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}\nRun scripts/prepare_data.py first.", file=sys.stderr)
        return 1
    env = environment_info()
    print(f"torch={env['torch']} cuda={env['cuda_available']} gpu={env['gpu']}")
    kwargs = {"device": args.device, "batch_size": args.batch_size} if args.extractor == "vgg19_fc7" else {}
    try:
        extractor = get_extractor(args.extractor, **kwargs)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = resolve_path(args.out) if args.out else processed / "features"
    summary = extract_split_features(records, info["images_root"], out, extractor, force=args.force)
    print(summary)
    print(f"features written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

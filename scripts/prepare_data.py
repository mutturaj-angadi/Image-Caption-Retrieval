"""Validate and prepare a dataset: writes metadata.json, vocab.json, stats.json, validation_report.json.

Examples
    # full historical split (Karpathy file; restval merged into train: 113,287 / 5,000 / 5,000)
    python scripts/prepare_data.py --source karpathy
    # labelled development subset
    python scripts/prepare_data.py --source karpathy --max-train 10000 --max-val 1000 --max-test 1000
    # tiny synthetic data for smoke tests (NOT COCO)
    python scripts/prepare_data.py --source synthetic
"""
import _bootstrap  # noqa: F401

import argparse
import sys

from src.data.prepare import prepare_dataset
from src.data.records import load_coco_captions, load_karpathy, make_random_split
from src.data.synthetic import make_synthetic_dataset
from src.utils.common import resolve_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["karpathy", "coco", "synthetic"], default="karpathy")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--karpathy-json", default=None, help="default: <raw-dir>/karpathy/dataset_coco.json")
    ap.add_argument("--images-root", default=None, help="default: <raw-dir>/coco")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--min-freq", type=int, default=5, help="minimum TRAIN word frequency for the vocabulary")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-image-check", action="store_true", help="skip opening image files (faster, less safe)")
    ap.add_argument("--deep-check", action="store_true", help="fully decode every image (slow)")
    ap.add_argument("--hash-duplicates", action="store_true", help="detect byte-identical images (slow)")
    args = ap.parse_args()

    raw = resolve_path(args.raw_dir)
    limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    try:
        if args.source == "karpathy":
            jp = resolve_path(args.karpathy_json) if args.karpathy_json else raw / "karpathy" / "dataset_coco.json"
            if not jp.is_file():
                print(f"Karpathy split file not found: {jp}\nRun: python scripts/download_data.py --what karpathy", file=sys.stderr)
                return 1
            records = load_karpathy(jp)
            root, name, split_src, synthetic = (resolve_path(args.images_root) if args.images_root else raw / "coco",
                                                "MS-COCO 2014 captions", "Karpathy & Fei-Fei split (restval merged into train)", False)
        elif args.source == "coco":
            ann = raw / "coco" / "annotations"
            records = []
            for part in ("train2014", "val2014"):
                p = ann / f"captions_{part}.json"
                if not p.is_file():
                    print(f"annotation file not found: {p}", file=sys.stderr)
                    return 1
                records += load_coco_captions(p, part)
            records = make_random_split(records, 5000, 5000, args.seed)
            root, name, split_src, synthetic = (resolve_path(args.images_root) if args.images_root else raw / "coco",
                                                "MS-COCO 2014 captions", f"random split (seed {args.seed}); NOT the historical split", False)
        else:
            records, root = make_synthetic_dataset(resolve_path("data/smoke/synthetic"))
            name, split_src, synthetic = "synthetic-shapes (SMOKE TEST ONLY)", "generated", True
            args.min_freq = 1
        res = prepare_dataset(records, root, resolve_path(args.out), dataset_name=name, split_source=split_src,
                              synthetic=synthetic, min_freq=args.min_freq, max_per_split=limits, seed=args.seed,
                              check_images=not args.no_image_check, deep_check=args.deep_check,
                              hash_duplicates=args.hash_duplicates)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    v = res["validation"]
    print(f"\nimages per split : {res['dataset']['num_images']}")
    print(f"vocabulary size  : {res['dataset']['vocab_size']}")
    print(f"missing / corrupt: {v['num_missing']} / {v['num_corrupted']}")
    print(f"subset           : {res['dataset']['is_subset']}  (label it as a subset in your report!)")
    print(f"written to       : {resolve_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

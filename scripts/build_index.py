"""Embed the images of a split with a trained model and save the retrieval index used by the web app.

    python scripts/build_index.py --config configs/gru.yaml                 # indexes the config's index.split (test)
    python scripts/build_index.py --config configs/gru.yaml --split val

Re-run after retraining or after adding images (extract their features first). The index records the
SHA-256 of the checkpoint it was built from.
"""
import _bootstrap  # noqa: F401

import argparse
import sys

from src.training.runner import build_run_index
from src.utils.common import load_config, parse_overrides


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (repeatable)")
    args = ap.parse_args()
    try:
        index = build_run_index(load_config(args.config, parse_overrides(args.set)), args.split, args.device)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"indexed {index.size} images, dim={index.dim}, backend={index.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

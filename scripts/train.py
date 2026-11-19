"""Train (or fit) a model described by a YAML config.

    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/gru.yaml
    python scripts/train.py --config configs/lstm.yaml --set train.epochs=10 --set train.batch_size=64
    python scripts/train.py --config configs/gru.yaml --resume          # continue from artifacts/<name>/last.pt

Model selection / early stopping use the VALIDATION split only. Artifacts: artifacts/<name>/
(best.pt, last.pt, history.json, vocab.json, model_metadata.json, train.log.jsonl).
"""
import _bootstrap  # noqa: F401

import argparse
import sys

from src.training.runner import train_model
from src.utils.common import deep_update, load_config, parse_overrides


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (repeatable)")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    try:
        overrides = parse_overrides(args.set)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.device:
        overrides = deep_update(overrides, {"train": {"device": args.device}})
    if args.resume:
        overrides = deep_update(overrides, {"train": {"resume": True}})
    try:
        cfg = load_config(args.config, overrides)
        meta = train_model(cfg)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\ntraining summary:", {k: meta["training"].get(k) for k in
                                  ("epochs_run", "best_epoch", "best_val_metric", "train_seconds", "device")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate a trained model and write reports/metrics/<name>_<split>_<protocol>.json.

    python scripts/evaluate.py --config configs/gru.yaml --split val
    python scripts/evaluate.py --config configs/gru.yaml --split test --protocol full --allow-test
    python scripts/evaluate.py --config configs/gru.yaml --split test --protocol 1k --allow-test   # needs >= 1000 test images

The test split must be requested explicitly (--allow-test) so it is not used by accident during model
selection. Report test numbers only after the configuration is final.
"""
import _bootstrap  # noqa: F401

import argparse
import sys

from src.evaluation.metrics import format_metrics
from src.training.runner import evaluate_run
from src.utils.common import load_config, parse_overrides


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--protocol", choices=["full", "1k"], default="full")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (repeatable)")
    ap.add_argument("--allow-test", action="store_true", help="required to evaluate on the test split")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        print("Refusing to evaluate on the TEST split without --allow-test (avoid selecting models on test data).", file=sys.stderr)
        return 2
    try:
        res = evaluate_run(load_config(args.config, parse_overrides(args.set)), args.split, args.protocol, device=args.device)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"\n{res['name']} [{args.split}, {args.protocol}, {res['num_images']} images / {res['num_captions']} captions]")
    print(format_metrics(res["metrics"]))
    print(f"query latency (median): {res['query_latency_ms_median']} ms | {res['latency_setup']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

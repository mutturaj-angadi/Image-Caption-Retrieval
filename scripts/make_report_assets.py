"""Create figures and the comparison table from REAL experiment outputs only.

Reads  reports/metrics/*_<split>_<protocol>.json  and  artifacts/<name>/history.json
Writes reports/figures/*.png and reports/metrics/summary.json (+ summary.md table)

Anything that has not been run is listed as "pending"; nothing is invented.
"""
import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

from src.utils.common import load_json, resolve_path, save_json

# Model order and display names for the comparison. Names must match configs/*.yaml "name".
MODELS = [("baseline_linreg", "Linear regression baseline"), ("gru_random_init", "GRU (random init)"),
          ("gru_glove", "GRU + GloVe"), ("lstm_glove", "LSTM + GloVe")]
METRICS = ["t2i_R@1", "t2i_R@5", "t2i_R@10", "t2i_mean_rank", "i2t_R@1", "i2t_R@5", "i2t_R@10", "i2t_mean_rank"]


def load_results(reports: Path, artifacts: Path, split: str, protocol: str) -> dict:
    out = {}
    for name, label in MODELS:
        p = reports / "metrics" / f"{name}_{split}_{protocol}.json"
        entry = {"label": label, "status": "pending"}
        if p.is_file():
            r = load_json(p)
            if r.get("dataset", {}).get("synthetic"):
                # Smoke-test output must never reach figures or tables.
                entry.update(status="pending", note="ignored: result was computed on SYNTHETIC smoke-test data")
                out[name] = entry
                continue
            entry.update(status="measured", metrics=r["metrics"], num_images=r["num_images"],
                         num_captions=r["num_captions"], latency_ms=r.get("query_latency_ms_median"),
                         train_seconds=r.get("training_seconds"), dataset=r.get("dataset", {}).get("name"),
                         synthetic=bool(r.get("dataset", {}).get("synthetic")), subset=bool(r.get("dataset", {}).get("is_subset")))
        h = artifacts / name / "history.json"
        if h.is_file():
            entry["history_file"] = h.as_posix()
        out[name] = entry
    return out


def markdown_table(results: dict) -> str:
    head = "| Model | T2I R@1 | T2I R@5 | T2I R@10 | T2I mean rank | I2T R@1 | I2T R@10 | Train time (s) | Latency (ms) |\n|---|---|---|---|---|---|---|---|---|\n"
    rows = []
    for e in results.values():
        if e["status"] != "measured":
            rows.append(f"| {e['label']} | pending | pending | pending | pending | pending | pending | pending | pending |")
            continue
        m = e["metrics"]
        rows.append(f"| {e['label']} | {m['t2i_R@1']:.1f} | {m['t2i_R@5']:.1f} | {m['t2i_R@10']:.1f} | {m['t2i_mean_rank']:.1f} | "
                    f"{m['i2t_R@1']:.1f} | {m['i2t_R@10']:.1f} | {e['train_seconds'] if e['train_seconds'] is not None else 'n/a'} | "
                    f"{e['latency_ms'] if e['latency_ms'] is not None else 'n/a'} |")
    return head + "\n".join(rows) + "\n"


def make_figures(results: dict, figures: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    written = []
    measured = {k: v for k, v in results.items() if v["status"] == "measured"}
    if measured:
        fig, ax = plt.subplots(figsize=(7, 4))
        ks = ["t2i_R@1", "t2i_R@5", "t2i_R@10"]
        width = 0.8 / len(measured)
        for i, (name, e) in enumerate(measured.items()):
            ax.bar([j + i * width for j in range(3)], [e["metrics"][k] for k in ks], width, label=e["label"])
        ax.set_xticks([j + 0.4 - width / 2 for j in range(3)])
        ax.set_xticklabels(["R@1", "R@5", "R@10"])
        ax.set_ylabel("Recall (%)")
        ax.set_title("Text-to-image retrieval (test split)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "model_comparison.png", dpi=160)
        plt.close(fig)
        written.append("model_comparison.png")
    for name, e in results.items():
        hp = e.get("history_file")
        if not hp or name == "baseline_linreg":
            continue
        hist = load_json(hp)
        if not isinstance(hist, list) or not hist:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, [h["train_loss"] for h in hist], label="train loss")
        ax.plot(ep, [h["val_loss"] for h in hist], label="validation loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("ranking loss")
        ax.set_title(f"{e['label']} - loss curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{name}_loss.png", dpi=160)
        plt.close(fig)
        written.append(f"{name}_loss.png")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--split", default="test")
    ap.add_argument("--protocol", default="full")
    args = ap.parse_args()
    reports, artifacts = resolve_path(args.reports_dir), resolve_path(args.artifacts_dir)
    results = load_results(reports, artifacts, args.split, args.protocol)
    figs = make_figures(results, reports / "figures")
    save_json({"split": args.split, "protocol": args.protocol, "results": results}, reports / "metrics" / "summary.json")
    table = markdown_table(results)
    (reports / "metrics" / "summary.md").write_text(table, encoding="utf-8")
    print(table)
    print("figures:", figs or "none (no measured results yet)")
    if all(v["status"] == "pending" for v in results.values()):
        print("NOTE: no evaluation results found - every entry is 'pending'.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

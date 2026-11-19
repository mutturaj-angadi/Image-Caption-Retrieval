"""Build a complete demo workspace from SYNTHETIC data (no download, no GPU) and verify retrieval.

    python scripts/smoke_test.py                 # writes data/smoke/ and prints how to run the web app on it
    python scripts/smoke_test.py --write-env     # also writes .env so the API uses the smoke workspace

GRU/LSTM models are trained too when PyTorch is installed; the NumPy baseline always runs.
All numbers produced here are for plumbing checks only - never report them as results.
"""
import _bootstrap  # noqa: F401

import argparse
import sys

from src.smoke import build_smoke_workspace, smoke_service
from src.utils.common import REPO_ROOT, resolve_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="data/smoke")
    ap.add_argument("--write-env", action="store_true")
    args = ap.parse_args()
    work = resolve_path(args.workdir)
    info = build_smoke_workspace(work)
    svc = smoke_service(info)
    models = [m for m in svc.list_models() if m["available"]]
    if not models:
        print("smoke test FAILED: no model became available", file=sys.stderr)
        return 1
    res = svc.search("a red circle on a white background", 5)
    assert res["num_results"] == 5, res
    rel = lambda p: p.resolve().relative_to(REPO_ROOT).as_posix() if str(p.resolve()).startswith(str(REPO_ROOT)) else str(p)  # noqa: E731
    env_lines = {
        "NCIR_ARTIFACTS_DIR": rel(info["artifacts"]),
        "NCIR_METADATA_PATH": rel(info["processed"] / "metadata.json"),
        "NCIR_REPORTS_DIR": rel(work / "reports"),
    }
    print("\nSMOKE TEST PASSED  (synthetic data - not COCO, numbers are meaningless)")
    print("models available:", [m["name"] for m in models])
    if info["neural_skipped"]:
        print("note: PyTorch not installed -> only the NumPy baseline was trained")
    print("\nTo use this workspace in the web app, set (PowerShell):")
    for k, v in env_lines.items():
        print(f'  $env:{k}="{v}"')
    if args.write_env:
        (REPO_ROOT / ".env").write_text("".join(f"{k}={v}\n" for k, v in env_lines.items()), encoding="utf-8")
        print("\nwrote .env (delete it to go back to the default artifacts/ and data/processed/ locations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

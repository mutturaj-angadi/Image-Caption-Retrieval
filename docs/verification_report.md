# Verification report

Run date: 2026-11-19. No real MS-COCO data, GloVe file, or deployed cloud URL was available. Synthetic metrics below are only evidence that the software pipeline runs; they are not research results.

## Passed

| Check | Command / method | Result |
|---|---|---|
| Clean Python environment | `py -3.11 -m venv .venv`; `.venv\Scripts\python.exe -m pip install --upgrade pip`; `.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` | Installed successfully on supported Python 3.11; torch 2.14.1+cpu and torchvision installed. Generated `requirements.lock.txt` from this exact verified environment for repeat installs. |
| Dependency consistency | `.venv\Scripts\python.exe -m pip check` | No broken requirements found. |
| Pinned Python requirements | `.venv\Scripts\python.exe -m pip install --dry-run -r requirements.lock.txt` | All pinned versions resolve and are already installed in the verified environment. |
| Full backend/ML suite in clean environment | `.venv\Scripts\python.exe -m pytest -q` | 94 passed, 1 skipped. The skip is optional FAISS parity because `faiss-cpu` is not installed. Re-run on 2026-11-19 with the workspace Python 3.11 environment; one Starlette/httpx deprecation warning. One non-failing Starlette/httpx deprecation warning was emitted by the latest resolved dependencies. |
| Synthetic pipeline in workspace Python 3.11 environment | `.venv\Scripts\python.exe scripts/smoke_test.py --workdir data/smoke` | Passed preparation, feature cache, baseline/GRU/LSTM training, validation/test evaluation, index creation, model loading, and known-query retrieval over generated shapes. These outputs are synthetic only. |
| CUDA availability | `.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"` | `torch 2.14.1+cpu`; CUDA unavailable. CPU neural training did run in the synthetic pipeline. |
| Python syntax/import compilation | `.venv\Scripts\python.exe -m compileall -q src backend scripts` | Passed. |
| Python static analysis | No Python linter or type checker is configured in the repository. | No lint/type command was available; compileall and pytest were run. |
| Frontend clean dependency install | In `frontend/`: `npm ci` | Passed; lockfile is present and npm reported zero vulnerabilities. |
| Frontend type check | `npm run typecheck` | Passed. |
| Frontend tests | `npm test` | 13 tests passed across 3 files, including Models and Evaluation pending/measured states and Search/API behavior. |
| Frontend production build | `npm run build` | Passed with Vite 8.3.5; output generated under ignored `frontend/dist/`. |
| Frontend dependency audit | `npm audit` | 0 vulnerabilities after upgrading Vite, Vitest, and Tailwind CSS; removed unused Autoprefixer config dependency. |
| Live FastAPI startup and endpoints | Started `.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8765` with the synthetic artifacts configured; used clean-environment HTTPX to call `/health`, `/api/models`, `/api/stats`, `/api/evaluation`, valid and invalid `/api/search`, and a returned `/api/images/{id}` URL. | Passed: ready health, three available smoke models, synthetic stats/evaluation, ranked search, PNG response, and consistent 422 invalid-query error. Process was stopped after verification. |
| API degraded readiness and validation/security cases | `tests/test_api_http.py` in pytest | Passed, including HTTP 503 with no usable models, request validation, CORS, and image path/id rejection. |

## Failures encountered and resolved

- The first pytest invocation used a Windows temp directory the host denied. Pytest now sets `--basetemp=.pytest-tmp` through `pytest.ini`; the standard command then passed.
- A first `npm ci` invocation was issued at the repository root, where there is no package lock. Rerunning from `frontend/` succeeded. Keep npm commands scoped to `frontend/`.
- The first frontend type check after upgrading to Vite 8 found the Vitest `test` config key was not in Vite's type. `vite.config.ts` now imports `defineConfig` from `vitest/config`; the final type check and build pass.
- Models/Evaluation page tests exposed duplicate `R@10` table header keys; the headers now distinguish text-to-image from image-to-text metrics. The pending-state test expectations were also aligned with the rendered accessible banner title. All 13 frontend tests pass.
- The initial frontend dependency tree had 13 reported advisories. Updated the build/test toolchain and lockfile; the final `npm audit` reports zero.

## Not run or still required

- VGG-19 feature extraction from actual images. It requires the real image tree and downloads pretrained torchvision weights on first use.
- MS-COCO acquisition/preparation, GloVe-backed training, real validation/test metrics, report asset generation from real results, or qualitative error analysis. `data/raw/` and `data/processed/` have no prepared research dataset, and `reports/metrics/` contains no measured results.
- FAISS comparison test; `faiss-cpu` is optional and absent. Exact NumPy search was exercised.
- GitHub publication: target repository was verified empty; local Git history initialized for first publication to `main`. Final commit and push confirmation are recorded after publication.
- Hosted CI execution, cloud deployment, deployed CORS, persistent-volume behavior, and remote health checks. There are no connected deployment credentials or URL. The workflow and deployment guide are present, but service deployment is not claimed.
- PDF and PowerPoint regeneration. Their existing files were preserved; the Markdown report source was corrected to avoid describing synthetic verification as real research results.

## Reproduction commands

From the repository root:

```powershell
python -m pytest -q
python scripts/smoke_test.py --workdir data/smoke
```

From `frontend/`:

```powershell
npm ci
npm run typecheck
npm test
npm run build
npm audit
```

For real results, follow the data and experiment commands in the root README. Do not copy synthetic outputs into the final research report.

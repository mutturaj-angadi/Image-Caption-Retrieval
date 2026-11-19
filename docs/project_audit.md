# NeuralSeek repository audit

Audit date: 2026-11-19. This report reflects the files present in this workspace. The workspace initially had no `.git` directory. The target GitHub repository was verified to be empty, so this publication creates its first local commit and does not replace remote history. No original reports, slides, datasets, checkpoints, or cache files were overwritten during the audit.

## Critical

No critical defect was confirmed in the inspected code. Real MS-COCO data, GloVe vectors, trained weights, and retrieval indexes are absent, so real-data behavior and model quality remain unverified.

## High

| Finding | Evidence | Change / recommendation | Verification |
|---|---|---|---|
| A stale index could be served after a checkpoint was replaced. | `src/retrieval/search.py` previously checked index dimension only; index metadata already records `checkpoint_sha256`. | Retrieval loading now verifies checkpoint SHA-256 against model metadata and the index hash. Rebuild an index after retraining. | Regression tests pass in the 91-test suite. |
| A corrupt or partially written index could be accepted without checking its metadata against the row files. | `src/retrieval/index.py` previously loaded `.npy`, ids, and metadata without checking the format, size, or dimension fields. | Validate index format, row count, and dimension; atomically replace embeddings on save. | Metadata mismatch regression test passes. |
| `/health` returned HTTP 200 when no model was usable. | `backend/main.py` reported `status="degraded"` but left HTTP status at 200, which can satisfy host readiness probes. | `/health` now returns 503 when zero usable models are loaded and 200 otherwise. | HTTP tests and live ASGI check pass for ready behavior; degraded HTTP behavior covered by test. |

## Medium

| Finding | Evidence | Change / recommendation | Verification |
|---|---|---|---|
| Source-control state could not be inspected. | No `.git` directory exists in the workspace, so `git status` and prior-change inspection are unavailable. | The repository was initialized on `main`; the specified GitHub remote was confirmed empty before publication. | Workspace contents were inspected directly; Git history remains unavailable. |
| Test fixtures used the host's default pytest temporary directory, which is inaccessible in this Windows environment. | Initial `python -m pytest -q`: fixture setup raised `PermissionError` under `%LOCALAPPDATA%\Temp`; tests that did not need `tmp_path` ran. | `pytest.ini` adds `--basetemp=.pytest-tmp` to pytest's supported `addopts`; the directory is ignored by Git. | The clean Python 3.11 run passes 94 tests, with one optional FAISS skip. |
| Dependency installation was not reproducible for the frontend. | `frontend/package.json` existed without a lockfile. | Add `package-lock.json`, use `npm ci`, and upgrade vulnerable Vite, Vitest, and Tailwind build tooling. | `npm ci` completed; typecheck, 13 Vitest tests, and production build pass; `npm audit` reports 0 vulnerabilities. |
| Python dependencies were specified only as ranges. | `requirements*.txt` allowed newer transitive package versions on each install. | Add `requirements.lock.txt` from the successful clean Python 3.11 environment; retain range files for supported dependency updates. | The locked versions match the clean environment that passed `pip check` and all 92 tests; CI and deployment use the lock. |
| No CI workflow was present. | No `.github/workflows/` files were found in the inspected tree. | Add CI for Python tests and frontend typecheck, tests, and production build without data/GPU downloads. | Workflow reviewed locally; hosted CI cannot be verified from this workspace. |
| Deployment steps and artifact persistence constraints were not collected in one guide. | README had local instructions; no `docs/deployment.md` existed. | Add a Render backend + Vercel static frontend guide, including environment, health, storage, and cold-start constraints. | Documentation reviewed; no cloud account or deployed URL is available for a live deployment check. |
| Search requests had stale-response protection but did not cancel network work. | `frontend/src/pages/SearchPage.tsx` ignored late responses but did not abort the previous fetch. | Pass an `AbortSignal`, cancel replaced searches, and abort on unmount. | API signal test and UI tests pass. |
| Some startup and malformed-artifact diagnostics included resolved local paths. | `src/retrieval/search.py`, `src/retrieval/models.py`, and `src/models/baseline.py` included exception strings or full file names in public model/health messages. | Replace these with stable actionable messages and keep detailed context in server logs. | Regression assertion checks missing-artifact health warnings do not disclose the configured path; final test run recorded in `docs/verification_report.md`. |
| Installation steps did not include a Unix-like shell path. | README setup focused on PowerShell. | Add Linux/macOS Python 3.11 virtual environment, locked install, and run commands. | Documentation-only change; commands were not executed on Linux/macOS. |

## Low

| Finding | Evidence | Change / recommendation | Verification |
|---|---|---|---|
| Pydantic response schemas used shared mutable list/dict defaults. | `backend/schemas.py` declared values such as `captions: list[str] = []`. | Replace with `Field(default_factory=...)`. | Covered by application import and API tests. |
| The runtime is outside the documented support range. | The current host is Python 3.13.5; `requirements.txt` documents Python 3.10–3.12. | Use Python 3.12 for supported local work and CI. Python 3.13 checks below are useful evidence but do not establish support. | Tests ran under 3.13.5; Python 3.12 was not available for a second runtime check. |

## Areas inspected

Inspected the repository tree, dependency and frontend package configuration, configs, data and feature pipeline, baseline and neural model code, trainer/checkpoints, evaluation, retrieval/indexing, scripts, backend routes/settings/schemas, React pages/components/API client, tests, `.gitignore`, README, architecture/methodology/troubleshooting/submission/traceability/demo docs, report source, and the artifact/data/report directories.

The repository has deliberate separation between preparation, training, evaluation, index construction, serving, and UI. Train-only vocabulary generation, shared text encoding, validation-based selection, test-split guard, finite-value checks, image-root containment, and synthetic-result labelling are already present. The verification run generated ignored smoke artifacts under `data/smoke/`; there are no real dataset files, real feature caches, or measured research metrics in the workspace.

## Verification limits

The initial test invocation could not create pytest temp directories because of host permissions; pytest now uses an ignored project-local temp directory. The full test suite also passed in a clean Python 3.11 environment. CUDA and FAISS are unavailable; real data and cloud credentials are absent. A passing synthetic smoke run demonstrates pipeline plumbing only and cannot establish research results. See `README.md`, `docs/verification_report.md`, and `docs/deployment.md` for commands and limits.

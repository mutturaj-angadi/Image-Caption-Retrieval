# Submission checklist

Nothing in this checklist has been done on your behalf: no GitHub repository was created or pushed.

## 0. Before anything else
- [ ] Read the assignment guidelines (UE24CS352A) again and compare with `docs/traceability.md` (rubric, deadline, file names, report length, demo format).
- [ ] The brief says the report is "one page" in one place and "two pages" in another. **Ask your faculty which applies**; `reports/final_report.md` is written for two pages and can be trimmed.
- [ ] Confirm whether the paper's methodology (`docs/methodology.md`) matches the PDF.

## 1. Experiments (nothing is "done" until these exist)
- [ ] Route B of the README completed (or a clearly labelled subset).
- [ ] `reports/metrics/*_test_*.json` exist for baseline, GRU, LSTM (+ ablation if used).
- [ ] `python scripts/make_report_assets.py` run; figures in `reports/figures/`.
- [ ] Qualitative examples and 3+ failure cases saved (screenshots from the app are fine) and discussed in the report.

## 2. Repository
- [ ] `pytest` passes locally (note which tests, if any, are skipped and why).
- [ ] `cd frontend; npm ci; npm run typecheck; npm test; npm run build` pass.
- [ ] Team names/SRNs filled in (`frontend/src/team.ts`, `reports/final_report.md`, README "Team").
- [ ] Choose and add a `LICENSE` (or keep the repo private).
- [ ] Screenshots added under `docs/images/` and linked in the README.
- [ ] README results table replaced with real numbers from `reports/metrics/summary.md`.
- [ ] No `.env`, datasets, checkpoints or `node_modules` committed (`git status`, `.gitignore` covers them).

## 3. Suggested commit sequence
```powershell
git init; git branch -M main
git add .gitignore .env.example requirements*.txt pytest.ini NOTICE.md configs; git commit -m "chore: project scaffold, configs and dependencies"
git add src/utils src/data scripts/prepare_data.py scripts/download_data.py scripts/extract_features.py tests/test_data.py tests/conftest.py; git commit -m "feat(data): dataset pipeline, validation, text and feature extraction"
git add src/models src/training scripts/train.py tests/test_models.py; git commit -m "feat(models): baseline, GRU/LSTM joint embedding, ranking loss, trainer"
git add src/evaluation src/retrieval src/smoke.py scripts/evaluate.py scripts/build_index.py scripts/smoke_test.py tests/test_retrieval.py; git commit -m "feat(retrieval): metrics, index, search service, smoke pipeline"
git add backend tests/test_api.py tests/test_api_http.py; git commit -m "feat(api): FastAPI backend"
git add frontend; git commit -m "feat(ui): React frontend"
git add docs reports presentation README.md notebooks scripts/make_report_assets.py scripts/build_report.py scripts/build_slides.py; git commit -m "docs: README, methodology, report, slides, viva"
```

## 4. Private GitHub repository and access for faculty/TAs
1. github.com → **New repository** → name `neural-caption-image-retrieval` → **Private** → do not add README/.gitignore (they already exist) → Create.
2. `git remote add origin https://github.com/<your-username>/neural-caption-image-retrieval.git`
3. `git push -u origin main` (sign in via the browser or a personal access token when asked).
4. Repository → **Settings → Collaborators → Add people** → enter the GitHub usernames of the faculty member and TAs (ask them for the usernames) → choose **Read** access.
5. Open the repository in a private browser window or ask a teammate to confirm they can clone it.
6. Submit the repository URL in the format the assignment requires.

## 5. Final acceptance (tick only what is true)
- [ ] Paper methodology reviewed and mapped · [ ] dataset pipeline works on real data · [ ] features extracted · [ ] baseline, GRU, LSTM trained · [ ] checkpoints load · [ ] retrieval returns correct images · [ ] metrics reported from real runs · [ ] API works · [ ] frontend works · [ ] tests run (list skips) · [ ] README reproducible on a clean machine · [ ] report + PDF · [ ] slides + demo rehearsed · [ ] viva answers read · [ ] limitations disclosed.

## 6. Deployment acceptance

- [ ] Backend artifacts, metadata, reports, and referenced images are stored on a persistent mounted volume; there is no reliance on ephemeral container storage.
- [ ] Backend `NCIR_*` variables use the mounted paths, and `NCIR_CORS_ORIGINS` contains the deployed frontend origin.
- [ ] The hosting platform's `/health` check returns HTTP 200 with at least one usable model after a restart or redeploy.
- [ ] Verify `/api/models`, `/api/stats`, `/api/evaluation`, valid search, invalid search, and an image URL returned by search on the deployed URL.
- [ ] Frontend production build succeeds with its public `VITE_API_BASE` set to the deployed backend origin; browser requests succeed without CORS errors.
- [ ] Cold-start time and memory fit the selected service plan, and the artifact volume remains available after restart.
- [ ] Record exact service URLs, deployment date, artifact version/hash, and verification output in the demo notes.

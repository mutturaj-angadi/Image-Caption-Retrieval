# Requirements → implementation → verification

Status legend: **Ran** = executed in the build environment; **Written** = implemented, to be verified on your machine; **Pending** = needs your data/experiments or documents.

**Important:** the three reference documents (paper, poster, assignment guidelines) were *not* available when this code was written.
Requirements below come from the project brief. Rows marked ⚠ must be re-checked against the assignment PDF (rubric, report length, file formats, deadlines, naming).

| # | Requirement | Implementation | Verification | Status |
|---|---|---|---|---|
| 1 | Load & validate image-caption data | `src/data/records.py`, `validate.py`, `prepare.py` | `tests/test_data.py` (missing, corrupt, duplicate ids/content, no captions, invalid split) | Ran |
| 2 | Official split, labelled subsets, no leakage | `load_karpathy`, `subset_records`, `check_no_leakage` | tests for restval merge, per-split deterministic subsets | Ran (logic); real COCO Pending |
| 3 | Smoke-test mode without COCO | `src/data/synthetic.py`, `src/smoke.py`, `scripts/smoke_test.py` | CLI chain + `smoke_ws` fixture | Ran |
| 4 | Text cleaning, vocab, padding/masks, GloVe init | `src/data/text.py`, `TextEncoder` | `test_data.py`, `test_models.py` (padding invariance), synthetic GRU/LSTM pipeline | Ran; real GloVe coverage pending |
| 5 | VGG-19 fc7 features, caching, shape checks | `src/data/features.py` | cache/alignment tests on the synthetic extractor | Cache Ran; VGG-19 not run (requires real images and pretrained weights) |
| 6 | Baseline: mean GloVe → linear regression → retrieval | `src/models/baseline.py` | recovers known linear map; chunked == materialised; persistence | Ran |
| 7 | GRU + GloVe, LSTM + GloVe | `text_encoder.py`, `joint_model.py` | shape, unit norm, frozen embeddings, checkpoint round-trip, synthetic training | Synthetic GRU/LSTM Ran; GloVe experiments pending |
| 8 | Ranking loss with valid negatives | `src/training/loss.py` | NumPy reference and torch == NumPy test | Ran |
| 9 | Config, optimiser, margin, checkpointing, early stopping, resume | `configs/*.yaml`, `trainer.py` | `EarlyStopper` tests; end-to-end synthetic training and checkpoint load | Synthetic pipeline Ran; real training pending |
| 10 | Recall@1/5/10, mean rank, both directions | `src/evaluation/metrics.py` | hand-built ranks, ties, perfect/random embeddings, `1k` folds | Ran |
| 11 | Index, normalisation, exact fallback, FAISS, rebuild | `src/retrieval/index.py` | brute-force equality, add/rebuild, persistence, metadata/hash validation; FAISS == exact | Exact Ran; FAISS skipped (package missing) |
| 12 | `search_images(query, top_k)` with validation and errors | `src/retrieval/search.py` | empty/blank/long query, bad `top_k`, unknown model, missing artifacts, missing files | Ran |
| 13 | ID → image file mapping correctness | `ImageStore`, index tests | row↔feature test; path-escape test | Ran |
| 14 | FastAPI endpoints + errors + CORS + safe images | `backend/main.py` | `tests/test_api.py`, `tests/test_api_http.py`, live local ASGI requests | Ran |
| 15 | React UI: Search, Results, Models, Evaluation, About | `frontend/src/pages/*` | Vitest tests, TypeScript check, production build | Ran (13 frontend tests) |
| 16 | Experiments: baseline, GRU, LSTM, ablation | `configs/`, `scripts/train.py`, `evaluate.py` | result JSON + `make_report_assets.py` | **Pending (your runs)** |
| 17 | Training curves, comparison chart, qualitative/failure analysis | `make_report_assets.py`, Evaluation page | generated from real files only; synthetic results are ignored | Written; data Pending |
| 18 | Reproducibility: seeds, config, metadata, env record | `set_seed`, `model_metadata.json` | metadata written in smoke run | Ran |
| 19 | README, Windows instructions, troubleshooting | `README.md`, `docs/` | selected Windows PowerShell commands and local API checks executed; full COCO setup pending | Partly Ran |
| 20 | Report (2 pages) + PDF | `reports/final_report.md`, `scripts/build_report.py` | PDF generated and rendered here | Ran (results Pending) |
| 21 | Presentation 8–10 slides + demo script | `presentation/`, `scripts/build_slides.py`, `docs/demo_script.md` | PPTX generated and rendered here | Ran (results Pending) |
| 22 | ≥ 40 viva questions | `docs/viva_qa.md` | count checked by script | Ran |
| 23 | GitHub submission checklist | `docs/submission_checklist.md` | — (no repository was created or pushed) | Written |
| 24 | Optional CLIP comparison | not implemented | — | Not done |
| 25 | Assignment-specific rules ⚠ | — | read the guidelines PDF; fill gaps | **Pending** |

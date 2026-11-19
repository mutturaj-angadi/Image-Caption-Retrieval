# Architecture

## Components

| Layer | Module | Responsibility |
|---|---|---|
| Data | `src/data/records.py`, `validate.py`, `splits.py`, `prepare.py` | load annotations, validate files, leakage check, subsets, statistics |
| Text | `src/data/text.py` | cleaning, tokenisation, vocabulary (train only), padding/lengths, GloVe matrix |
| Features | `src/data/features.py` | VGG-19 fc7 extraction (torchvision) with an on-disk cache aligned to image ids |
| Models | `src/models/baseline.py`, `text_encoder.py`, `joint_model.py` | ridge regression baseline; GRU/LSTM text encoder; joint model + checkpoints |
| Training | `src/training/loss.py`, `trainer.py`, `runner.py` | ranking loss, loop, early stopping, orchestration, metadata |
| Evaluation | `src/evaluation/metrics.py` | Recall@K, mean/median rank, protocols (`full`, `1k`) |
| Retrieval | `src/retrieval/index.py`, `models.py`, `search.py` | normalised embedding index (exact / FAISS), retrievers, `search_images()` |
| API | `backend/` | FastAPI app, Pydantic schemas, evaluation/stat readers |
| UI | `frontend/` | React pages: Search, Models, Evaluation, About |

## Data flow

1. `prepare_data.py` → `metadata.json`, `vocab.json`, `stats.json`.
2. `extract_features.py` → `features/<split>_features.npy` (+ ids). Computed once, reused by every model.
3. `train.py` → `artifacts/<name>/{best.pt|baseline.npz, last.pt, history.json, vocab.json, model_metadata.json}`.
4. `evaluate.py` → `reports/metrics/<name>_<split>_<protocol>.json` and updates `model_metadata.json`.
5. `build_index.py` → `artifacts/<name>/index/{embeddings.npy, ids.json, index_meta.json}`.
6. `uvicorn backend.main:app` loads every model folder **once** at startup into `SearchService`.
7. A query is validated, tokenised with the model's vocabulary, encoded, normalised, searched in the index and returned with captions from the metadata.

## Design decisions

* **One code path for encoding.** Evaluation, index building and the web search all call the same `retriever.encode_captions / encode_image_features`, so they cannot disagree on preprocessing.
* **Features are cached.** VGG-19 is frozen (as in the research setup), so image features are extracted once; training only touches small projection layers and the text encoder.
* **Graceful degradation.** Missing artifacts never crash startup: each model is reported `available=false` with a reason; `/health` returns `degraded` and HTTP 503 until at least one usable model and index are loaded.
* **Safe file access.** Images are served only for ids present in `metadata.json`, resolved under a fixed root, never from user-supplied paths.
* **No extra infrastructure.** No database, Docker or cloud service is needed; everything is files + one Python process + one static frontend.
* **Exact search by default.** Brute-force cosine search over ≤ 10⁵ vectors is fast enough (N×D multiply-adds per query); FAISS `IndexFlatIP` is used only above `index.faiss_min_size` when installed, and gives identical results.

## Artifacts and metadata

`model_metadata.json` records architecture, embedding size, training configuration, word-vector source and coverage, dataset description (incl. subset/synthetic flags), checkpoint file + SHA-256, training summary, environment (Python/torch/GPU) and every evaluation run. The index stores the checkpoint hash it was built from.

## Deployment notes

Single-process FastAPI is sufficient for a demonstration. Request handlers are synchronous (CPU-bound) and run in Starlette's thread pool; the model is shared and read-only. For production one would add authentication, rate limiting, an ANN index for very large corpora and image-serving through a CDN.

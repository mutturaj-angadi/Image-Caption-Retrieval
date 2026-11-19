# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `.\.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, reopen PowerShell |
| `python` opens the Microsoft Store | App-execution alias | use `py -3.12`, or disable the alias in *Settings → Apps → Advanced app settings → App execution aliases* |
| `pip install torch` is huge / slow | CUDA wheels | CPU-only is fine: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| `error: PyTorch is not installed, but GRU/LSTM training needs it` | torch missing in the active venv | activate `.venv`, install `requirements.lock.txt` or choose a matching GPU build as described in the README |
| `CUDA out of memory` during feature extraction | batch too large | `--batch-size 8` (or 4); or `--device cpu` |
| Feature extraction takes hours | VGG-19 on CPU | use a subset (`--max-train 20000`), a GPU, or Google Colab; features are cached so it is a one-time cost |
| `GloVe file not found` | not downloaded | `python scripts/download_data.py --what glove` or place `glove.6B.300d.txt` in `data/raw/glove/` |
| `Karpathy split file not found` | not downloaded | see `data/README.md` |
| `MemoryError` while loading features | full 123k × 4096 float32 ≈ 2 GB per copy | use a subset, close other apps, or train on fewer images |
| Training loss stays at the margin value | learning rate too low/high, or collapsed embeddings | try `--set train.lr=0.0005`, `--set train.margin=0.1`, `--set train.max_violation=false`; check `history.json` |
| Validation metric never improves | too small subset, or data mismatch | verify `validation_report.json`, vocabulary coverage in `model_metadata.json → word_vectors` |
| `evaluate.py` says *Refusing to evaluate on the TEST split* | safety guard | add `--allow-test` only when the configuration is final |
| `1k protocol needs at least 1000 images` | small test subset | use `--protocol full` |
| API `/health` says `degraded` | no usable model | `warnings` in the response says why; train, evaluate and run `build_index.py` |
| `model ... is not available: missing file ... best.pt` | model folder incomplete | re-run `scripts/train.py`; for the baseline the file is `baseline.npz` |
| `index dim ... != model embedding dim ...; rebuild the index` | retrained with different dimensions | `python scripts/build_index.py --config configs/<model>.yaml` |
| Search says *none of the words in the query are in the model vocabulary* | query uses only unseen words | rephrase; the synthetic demo only knows colours/shapes |
| Images show "Image file unavailable" | `images_root` in metadata differs from your disk | set `NCIR_IMAGES_ROOT` in `.env` to the folder that contains `train2014/` and `val2014/` |
| Browser shows *Cannot reach the API* | backend not running or wrong port | start `uvicorn backend.main:app --port 8000`; the dev frontend proxies `/api` to port 8000 |
| CORS error when calling the API from another origin | origin not allowed | add it to `NCIR_CORS_ORIGINS` |
| `npm ci` fails on a version | registry/cache problem or unsupported Node version | use Node.js 20.19+ or 22.12+, then run `npm ci`; keep the committed lockfile |
| `faiss-cpu` fails to install | no wheel for your Python | skip it; exact search is used automatically |
| Tests skipped | optional packages missing | install `requirements-dev.txt` (+ torch) to run them |
| Port 5173/8000 already in use | another process | `uvicorn ... --port 8001` and set `VITE_API_BASE=http://127.0.0.1:8001` (or change the proxy in `vite.config.ts`) |

## Useful checks

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/smoke_test.py          # verifies the whole pipeline in a minute
pytest -q -rs                         # -rs lists why tests were skipped
Invoke-RestMethod http://localhost:8000/health
```

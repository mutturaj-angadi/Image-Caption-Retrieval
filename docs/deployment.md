# Deployment guide

This repository can deploy as a static Vite frontend on Vercel and a FastAPI web service on Render. Deployment has not been performed from this workspace. A service is not ready to serve search until trained artifacts, an index, metadata, and image files are available to it.

## 1. Prepare a real artifact bundle

Run the dataset, feature, training, evaluation, and index commands in the root README on a machine with MS-COCO 2014 data and the GloVe file. Keep `artifacts/<model>/` and `data/processed/metadata.json` together with the referenced image tree. Do not commit them to Git. Record the output of the commands and label any subset or synthetic run accurately.

The backend currently reads from local paths. On Render, attach persistent storage and copy the trained artifacts, metadata, and image tree onto that disk. The disk contents must survive deploys and be mounted at a stable absolute path. Object storage is not wired into `ImageStore`; using a bucket requires an explicit storage adapter or a mounted/object-backed filesystem. Do not rely on the container's ephemeral filesystem for these files.

## 2. Deploy the backend on Render

Create a **Web Service** from the repository. Set the repository root as the service root, use Python 3.12, and configure:

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.lock.txt` |
| Start command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |
| Persistent disk mount | for example `/var/data/neuralseek` |

Set these environment variables in the service dashboard (adapt paths to the mounted disk):

```text
NCIR_ARTIFACTS_DIR=/var/data/neuralseek/artifacts
NCIR_METADATA_PATH=/var/data/neuralseek/processed/metadata.json
NCIR_IMAGES_ROOT=/var/data/neuralseek/coco
NCIR_REPORTS_DIR=/var/data/neuralseek/reports
NCIR_STATS_PATH=/var/data/neuralseek/processed/stats.json
NCIR_CORS_ORIGINS=https://YOUR-VERCEL-DOMAIN
NCIR_MAX_TOP_K=100
NCIR_MAX_QUERY_CHARS=300
NCIR_LOG_LEVEL=INFO
```

Do not put secrets in the repository; these settings do not require secrets. `/health` returns 200 only when at least one model and index pass compatibility checks, and 503 when the app is degraded. Inspect Render logs for startup warnings and then verify:

```powershell
Invoke-RestMethod https://YOUR-RENDER-SERVICE.onrender.com/health
Invoke-RestMethod https://YOUR-RENDER-SERVICE.onrender.com/api/models
Invoke-RestMethod https://YOUR-RENDER-SERVICE.onrender.com/api/evaluation
Invoke-RestMethod -Method Post -Uri https://YOUR-RENDER-SERVICE.onrender.com/api/search `
  -ContentType 'application/json' -Body '{"query":"a dog running through snow","top_k":5}'
```

The first request after an idle period may be slow if the hosting plan suspends the instance. Model loading and index loading happen at process startup, so allow for their RAM use and startup time. The full COCO feature cache alone is roughly 2 GB; the images are roughly 19 GB. Use a development subset for a small demonstration and size persistent storage and memory for the actual artifacts. Multiple service instances each need access to the same artifact version; this code does not synchronize index or checkpoint updates at runtime.

## 3. Deploy the frontend on Vercel

Import the same repository and set the Vercel **Root Directory** to `frontend`.

| Setting | Value |
|---|---|
| Framework preset | Vite |
| Install command | `npm ci` |
| Build command | `npm run build` |
| Output directory | `dist` |

Set the frontend environment variable:

```text
VITE_API_BASE=https://YOUR-RENDER-SERVICE.onrender.com
```

Vite embeds `VITE_*` values in the public JavaScript bundle. This value is a public API origin, never a secret. After the frontend URL is assigned, add that exact origin to backend `NCIR_CORS_ORIGINS`, redeploy the backend, then redeploy the frontend. Verify a search from the deployed browser and confirm the browser console has no CORS or mixed-content errors. The app uses hash-based navigation, so client-side page refreshes do not need special path rewrites.

## 4. Local production-build check

From PowerShell:

```powershell
cd frontend
npm ci
$env:VITE_API_BASE = 'http://127.0.0.1:8000'
npm run build
Remove-Item Env:VITE_API_BASE
```

Then run the backend from the repository root with the same `NCIR_*` paths and verify the local health and search requests above. A successful frontend build checks bundling and types; it does not prove that a hosted backend or CORS configuration is reachable.

## Platform references

- [Render deploys](https://render.com/docs/deploys), [Render web services](https://render.com/docs/web-services), and [Render persistent disks](https://render.com/docs/disks).
- [Vite static deployment guide](https://vite.dev/guide/static-deploy.html).

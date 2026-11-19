# Data directory

Nothing in `raw/` or `processed/` is committed to Git (see `.gitignore`).

## Expected layout after downloading

```
data/raw/
  karpathy/dataset_coco.json            # Karpathy & Fei-Fei split + 5 captions per image
  glove/glove.6B.300d.txt               # GloVe 6B, 300-d
  coco/
    annotations/captions_train2014.json # only needed for --source coco
    annotations/captions_val2014.json
    train2014/COCO_train2014_*.jpg      # 82,783 images
    val2014/COCO_val2014_*.jpg          # 40,504 images
```

`filepath` entries in `dataset_coco.json` are `train2014` / `val2014`, so the images root is `data/raw/coco`.

## Sizes

| Item | Approx. size |
|---|---|
| train2014 + val2014 images | ~19 GB |
| annotations | ~250 MB |
| Karpathy split file | ~40 MB |
| GloVe 6B zip (only the 300-d file is extracted) | ~860 MB zip, ~1 GB extracted |
| Cached VGG-19 fc7 features, full split | 123k images × 4096 × 4 bytes ≈ 2 GB |

## Manual download (if the script is blocked by your network)

1. COCO 2014 images and annotations: https://cocodataset.org/#download
2. Karpathy split (`caption_datasets.zip`, take `dataset_coco.json`): https://cs.stanford.edu/people/karpathy/deepimagesent/
3. GloVe: https://nlp.stanford.edu/projects/glove/ (`glove.6B.zip`, take `glove.6B.300d.txt`)

Place the files as shown above, then run `python scripts/prepare_data.py --source karpathy`.

## Generated files (`data/processed/`)

`metadata.json` (records + dataset description), `vocab.json` (built from TRAIN captions only), `stats.json`,
`validation_report.json`, `features/{train,val,test}_features.npy` + `*_ids.json` + `extractor.json`.

## Subsets and splits

* `--max-train/--max-val/--max-test` keep a deterministic random subset per split (seeded); splits are never mixed, so no leakage is introduced.
  Subset runs are flagged `is_subset: true` and shown as such in the UI and reports.
* `--source karpathy` reproduces the historical 113,287 / 5,000 / 5,000 split (restval merged into train).
* `--source coco` creates a *random* 5k/5k split with a seed; this is **not** the historical split.
* `--source synthetic` writes procedurally generated shapes to `data/smoke/` for smoke tests only.

## Alternative dataset

If COCO is unavailable, Flickr30k (31,783 images, 5 captions each, Karpathy split) is the natural substitute: write a loader that returns `ImageRecord`s
(see `src/data/records.py`). It is ~6× smaller, so absolute recalls are not comparable with COCO numbers.

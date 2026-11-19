"""End-to-end dataset preparation: validate, subset, split-check, vocabulary, statistics."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from src.data.records import ImageRecord, standardize_captions
from src.data.splits import group_by_split, subset_records
from src.data.text import Vocabulary, tokenize
from src.data.validate import check_no_leakage, validate_records
from src.utils.common import get_logger, save_json

log = get_logger("ncir.prepare")


def dataset_stats(records: list[ImageRecord], vocab: Optional[Vocabulary] = None) -> dict:
    """Split sizes, caption-length statistics and vocabulary coverage."""
    stats: dict = {"splits": {}}
    groups = group_by_split(records)
    for split, recs in groups.items():
        lens = np.asarray([len(tokenize(c)) for r in recs for c in r.captions], dtype=np.int64)
        entry = {"num_images": len(recs), "num_captions": int(lens.size)}
        if lens.size:
            entry.update({
                "caption_len_mean": float(lens.mean()), "caption_len_median": float(np.median(lens)),
                "caption_len_p95": float(np.percentile(lens, 95)), "caption_len_max": int(lens.max()),
                "caption_len_hist": np.bincount(np.minimum(lens, 40), minlength=41).tolist(),
            })
        if vocab is not None and lens.size:
            toks = [t for r in recs for c in r.captions for t in tokenize(c)]
            entry["oov_rate"] = float(sum(1 for t in toks if t not in vocab.stoi) / max(1, len(toks)))
        stats["splits"][split] = entry
    stats["total_images"] = len(records)
    stats["total_captions"] = sum(s["num_captions"] for s in stats["splits"].values())
    if vocab is not None:
        stats["vocab_size"] = len(vocab)
    return stats


def prepare_dataset(records: list[ImageRecord], images_root: str | Path, out_dir: str | Path, *,
                    dataset_name: str, split_source: str, synthetic: bool = False,
                    min_freq: int = 5, captions_per_image: Optional[int] = 5,
                    max_per_split: Optional[Mapping[str, Optional[int]]] = None, seed: int = 42,
                    check_images: bool = True, deep_check: bool = False, hash_duplicates: bool = False) -> dict:
    """Validate and prepare a dataset. Writes metadata.json, vocab.json, stats.json, validation_report.json.

    The vocabulary is built from TRAIN captions only. Subsetting never mixes splits.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, cap_report = standardize_captions(records, captions_per_image)
    valid, report = validate_records(records, images_root, check_images=check_images, deep=deep_check,
                                     hash_duplicates=hash_duplicates)
    report.update(cap_report)
    is_subset = bool(max_per_split) and any(v is not None for v in max_per_split.values())
    if is_subset:
        valid = subset_records(valid, max_per_split, seed=seed)
    check_no_leakage(valid)
    valid.sort(key=lambda r: (r.split, r.image_id))

    train_caps = [tokenize(c) for r in valid if r.split == "train" for c in r.captions]
    if not train_caps:
        raise ValueError("no training captions after validation - cannot build a vocabulary")
    vocab = Vocabulary.build(train_caps, min_freq=min_freq)
    stats = dataset_stats(valid, vocab)

    info = {
        "name": dataset_name, "split_source": split_source, "synthetic": synthetic, "is_subset": is_subset,
        "subset_limits": dict(max_per_split) if is_subset else None, "seed": seed,
        "captions_per_image": captions_per_image, "vocab_min_freq": min_freq, "vocab_size": len(vocab),
        "num_images": {s: len(v) for s, v in group_by_split(valid).items()},
    }
    save_json({"format": "ncir-metadata-v1", "dataset": info, "images_root": Path(images_root).as_posix(),
               "records": [r.to_dict() for r in valid]}, out / "metadata.json")
    vocab.save(out / "vocab.json")
    save_json(stats, out / "stats.json")
    save_json(report, out / "validation_report.json")
    log.info("prepared %s: %s (vocab=%d)%s", dataset_name, info["num_images"], len(vocab),
             " [SUBSET]" if is_subset else "")
    return {"dataset": info, "stats": stats, "validation": report}

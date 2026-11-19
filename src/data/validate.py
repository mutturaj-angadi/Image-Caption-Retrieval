"""Dataset validation: missing files, corrupted images, duplicates and split leakage."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image

from src.data.records import ImageRecord, VALID_SPLITS
from src.utils.common import progress

# Priority when a byte-identical image appears in several splits: keep the evaluation copy.
_KEEP_PRIORITY = {"test": 0, "val": 1, "train": 2}


class DataLeakageError(ValueError):
    """Raised when the same image id appears in more than one split."""


def check_image_file(path: Path, deep: bool = False) -> Optional[str]:
    """Return ``None`` if the file is a readable image, otherwise a short error string."""
    if not path.is_file():
        return "missing"
    try:
        with Image.open(path) as im:
            im.verify()  # header / structure check (cheap)
        if deep:
            with Image.open(path) as im:
                im.convert("RGB").load()  # full decode (slow, catches truncated files)
    except Exception as exc:  # noqa: BLE001 - PIL raises many different exception types
        return f"corrupted: {type(exc).__name__}"
    return None


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def validate_records(records: list[ImageRecord], images_root: str | Path, check_images: bool = True,
                     deep: bool = False, hash_duplicates: bool = False,
                     drop_duplicates: bool = True) -> tuple[list[ImageRecord], dict]:
    """Validate records against the files on disk.

    Removes records with missing / corrupted images, duplicate ids, invalid splits or no
    captions, and (optionally) byte-identical duplicate images. Returns ``(valid, report)``.
    """
    root = Path(images_root)
    report: dict = {
        "total_records": len(records), "missing_files": [], "corrupted_files": [],
        "duplicate_image_ids": [], "invalid_split": [], "no_captions": [],
        "duplicate_content_groups": [], "duplicates_dropped": [],
    }
    seen_ids: set[str] = set()
    valid: list[ImageRecord] = []
    for rec in progress(records, total=len(records), desc="validating"):
        if rec.split not in VALID_SPLITS:
            report["invalid_split"].append(rec.image_id)
            continue
        if rec.image_id in seen_ids:
            report["duplicate_image_ids"].append(rec.image_id)
            continue
        if not rec.captions:
            report["no_captions"].append(rec.image_id)
            continue
        if check_images:
            err = check_image_file(root / rec.file_name, deep=deep)
            if err == "missing":
                report["missing_files"].append(rec.file_name)
                continue
            if err:
                report["corrupted_files"].append({"file": rec.file_name, "error": err})
                continue
        seen_ids.add(rec.image_id)
        valid.append(rec)

    if hash_duplicates and check_images:
        groups: dict[str, list[ImageRecord]] = {}
        for rec in progress(valid, total=len(valid), desc="hashing"):
            groups.setdefault(file_md5(root / rec.file_name), []).append(rec)
        drop_ids: set[str] = set()
        for digest, members in groups.items():
            if len(members) < 2:
                continue
            members = sorted(members, key=lambda r: (_KEEP_PRIORITY[r.split], r.image_id))
            report["duplicate_content_groups"].append({"md5": digest, "image_ids": [m.image_id for m in members],
                                                       "splits": [m.split for m in members]})
            if drop_duplicates:
                drop_ids.update(m.image_id for m in members[1:])
        valid = [r for r in valid if r.image_id not in drop_ids]
        report["duplicates_dropped"] = sorted(drop_ids)

    report["valid_records"] = len(valid)
    report["num_missing"] = len(report["missing_files"])
    report["num_corrupted"] = len(report["corrupted_files"])
    # keep the JSON report small
    for key in ("missing_files", "corrupted_files", "duplicate_image_ids", "no_captions", "duplicates_dropped"):
        report[key + "_sample"] = report[key][:20]
        del report[key]
    return valid, report


def check_no_leakage(records: list[ImageRecord]) -> None:
    """Raise :class:`DataLeakageError` if an image id occurs in more than one split."""
    owner: dict[str, str] = {}
    for rec in records:
        prev = owner.setdefault(rec.image_id, rec.split)
        if prev != rec.split:
            raise DataLeakageError(f"image {rec.image_id} appears in both '{prev}' and '{rec.split}'")
    counts = sum(1 for _ in owner)
    if counts != len(records):
        raise DataLeakageError("duplicate image ids inside the record list")

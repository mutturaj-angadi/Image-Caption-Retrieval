"""Shared helpers: paths, config loading, seeding, logging, JSON IO and environment info."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- paths
def resolve_path(path: str | os.PathLike, base: Optional[Path] = None) -> Path:
    """Resolve ``path`` against the repository root (or ``base``) unless it is absolute.

    Using the repository root instead of the current directory keeps scripts, tests and the
    API consistent no matter where they are launched from.
    """
    p = Path(path)
    return p if p.is_absolute() else (base or REPO_ROOT) / p


# --------------------------------------------------------------------------- config
def deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict where ``override`` is recursively merged into ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | os.PathLike, overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Load a YAML config file and optionally merge ``overrides`` on top."""
    cfg_path = resolve_path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {cfg_path} must contain a YAML mapping at the top level")
    cfg = deep_update(cfg, overrides) if overrides else cfg
    for key in ("name", "model"):
        if key not in cfg:
            raise ValueError(f"Config {cfg_path} is missing required key '{key}'")
    return cfg


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Turn ``["train.epochs=5", "data.glove_path=null"]`` into a nested dict (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


# --------------------------------------------------------------------------- JSON
def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> Path:
    """Atomically write ``obj`` as UTF-8 JSON (write to a temp file, then replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return target


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- seeding
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # noqa: WPS433 - optional dependency

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- logging
class JsonLineFormatter(logging.Formatter):
    """Structured (JSON lines) log formatter used for the log file."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "ncir", log_file: str | os.PathLike | None = None,
               level: str | None = None) -> logging.Logger:
    """Return a logger with a console handler and an optional JSON-lines file handler."""
    logger = logging.getLogger(name)
    level_name = (level or os.environ.get("NCIR_LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    logger.propagate = False
    if not any(getattr(h, "_ncir_console", False) for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        console._ncir_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    if log_file is not None:
        path = Path(log_file).resolve()
        if not any(getattr(h, "_ncir_file", None) == path for h in logger.handlers):
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(JsonLineFormatter())
            fh._ncir_file = path  # type: ignore[attr-defined]
            logger.addHandler(fh)
    return logger


def progress(iterable: Iterable, total: Optional[int] = None, desc: str = "") -> Iterator:
    """Wrap ``iterable`` in a tqdm bar when tqdm is installed; otherwise pass it through."""
    try:
        from tqdm import tqdm

        return iter(tqdm(iterable, total=total, desc=desc, leave=False))
    except ImportError:
        return iter(iterable)


# --------------------------------------------------------------------------- environment
def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except ImportError:
        return False


def environment_info() -> dict[str, Any]:
    """Record the software / hardware environment of a run (for reproducibility)."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": None,
        "torchvision": None,
        "cuda_available": False,
        "gpu": None,
        "faiss": faiss_available(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
        try:
            import torchvision

            info["torchvision"] = torchvision.__version__
        except ImportError:
            pass
    except ImportError:
        pass
    return info

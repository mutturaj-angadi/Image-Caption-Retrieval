"""Backend settings read from environment variables (and an optional ``.env`` file at the repo root)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.common import REPO_ROOT, resolve_path


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Minimal ``.env`` loader (KEY=VALUE lines). Existing environment variables win. No extra dependency."""
    env_file = path or REPO_ROOT / ".env"
    loaded: dict[str, str] = {}
    if not env_file.is_file():
        return loaded
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


@dataclass
class Settings:
    artifacts_dir: Path
    metadata_path: Path
    images_root: Path | None
    reports_dir: Path
    stats_path: Path
    cors_origins: list[str] = field(default_factory=list)
    max_top_k: int = 100
    max_query_chars: int = 300
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        get = os.environ.get
        metadata = resolve_path(get("NCIR_METADATA_PATH", "data/processed/metadata.json"))
        stats = get("NCIR_STATS_PATH")
        root = get("NCIR_IMAGES_ROOT")
        return cls(
            artifacts_dir=resolve_path(get("NCIR_ARTIFACTS_DIR", "artifacts")),
            metadata_path=metadata,
            images_root=resolve_path(root) if root else None,
            reports_dir=resolve_path(get("NCIR_REPORTS_DIR", "reports")),
            stats_path=resolve_path(stats) if stats else metadata.parent / "stats.json",
            cors_origins=[o.strip() for o in get("NCIR_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()],
            max_top_k=int(get("NCIR_MAX_TOP_K", "100")),
            max_query_chars=int(get("NCIR_MAX_QUERY_CHARS", "300")),
            log_level=get("NCIR_LOG_LEVEL", "INFO"),
        )

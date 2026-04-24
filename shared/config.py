from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def get_config_value(*keys: str, default: Any = None) -> Any:
    value: Any = load_config()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path

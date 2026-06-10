from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_training_config(
    *,
    default_config: dict[str, Any],
    default_config_path: Path,
    description: str,
) -> tuple[dict[str, Any], Path]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help="Path to a YAML config file.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    if not isinstance(user_config, dict):
        raise ValueError(f"Config file must contain a YAML mapping at the root: {config_path}")

    return _deep_merge_dicts(default_config, user_config), config_path

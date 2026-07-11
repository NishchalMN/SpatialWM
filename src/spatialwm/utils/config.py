"""YAML config loading + key validation."""

from __future__ import annotations

from typing import Iterable


def load_config(path: str) -> dict:
    """Load a YAML file into a dict via ``yaml.safe_load``."""
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path!r} did not parse to a dict.")
    return cfg


def require_keys(cfg: dict, keys: Iterable[str]) -> None:
    """Raise ``KeyError`` listing any top-level keys missing from ``cfg``."""
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

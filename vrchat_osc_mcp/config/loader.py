from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .settings import Settings


@dataclass(frozen=True)
class LoadedSettings:
    settings: Settings
    config_path: Path | None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _detect_default_config_path(project_root: Path) -> Path | None:
    candidates = [project_root / "config.yaml", project_root / "config" / "config.yaml"]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def load_settings(*, project_root: Path, config_path: Path | None, cli_overrides: dict[str, Any]) -> LoadedSettings:
    resolved_path = config_path or _detect_default_config_path(project_root)

    yaml_data: dict[str, Any] = {}
    if resolved_path is not None and resolved_path.exists():
        raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            yaml_data = raw

    merged = _deep_merge(yaml_data, cli_overrides)
    settings = Settings.model_validate(merged)
    return LoadedSettings(settings=settings, config_path=resolved_path)

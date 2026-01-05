from __future__ import annotations

from pathlib import Path


def default_vrchat_osc_root() -> Path:
    # Windows default: C:\Users\<you>\AppData\LocalLow\VRChat\VRChat\OSC
    return Path.home() / "AppData" / "LocalLow" / "VRChat" / "VRChat" / "OSC"


def resolve_avatar_config_path(*, osc_root: Path | None, explicit_path: Path | None) -> Path | None:
    """Resolve an avatar config json path.

    Priority:
    1) explicit_path (CLI/YAML)
    2) newest avtr_*.json under <osc_root>/usr_*/Avatars/
    """

    if explicit_path is not None:
        p = explicit_path.expanduser()
        return p if p.exists() and p.is_file() else None

    root = (osc_root or default_vrchat_osc_root()).expanduser()
    if not root.exists():
        return None

    candidates: list[Path] = []
    for usr_dir in root.glob("usr_*"):
        avatars_dir = usr_dir / "Avatars"
        if not avatars_dir.exists():
            continue
        candidates.extend(avatars_dir.glob("avtr_*.json"))

    if not candidates:
        return None

    # Choose the most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)

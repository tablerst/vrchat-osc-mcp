from __future__ import annotations

import os
from pathlib import Path

from vrchat_osc_mcp.vrc_config.resolver import resolve_avatar_config_path_for_avatar_id


def test_resolve_avatar_config_path_for_avatar_id_finds_exact_file(tmp_path: Path):
    osc_root = tmp_path / "OSC"
    p = osc_root / "usr_123" / "Avatars" / "avtr_test.json"
    p.parent.mkdir(parents=True)
    p.write_text("{}", encoding="utf-8")

    resolved = resolve_avatar_config_path_for_avatar_id(osc_root=osc_root, avatar_id="avtr_test")
    assert resolved == p


def test_resolve_avatar_config_path_for_avatar_id_picks_newest_when_multiple(tmp_path: Path):
    osc_root = tmp_path / "OSC"

    p1 = osc_root / "usr_a" / "Avatars" / "avtr_same.json"
    p1.parent.mkdir(parents=True)
    p1.write_text("{}", encoding="utf-8")

    p2 = osc_root / "usr_b" / "Avatars" / "avtr_same.json"
    p2.parent.mkdir(parents=True)
    p2.write_text("{}", encoding="utf-8")

    # Make p2 appear newer deterministically.
    os.utime(p1, (1, 1))
    os.utime(p2, (2, 2))

    resolved = resolve_avatar_config_path_for_avatar_id(osc_root=osc_root, avatar_id="avtr_same")
    assert resolved == p2


def test_resolve_avatar_config_path_for_avatar_id_accepts_json_suffix(tmp_path: Path):
    osc_root = tmp_path / "OSC"
    p = osc_root / "usr_123" / "Avatars" / "avtr_test.json"
    p.parent.mkdir(parents=True)
    p.write_text("{}", encoding="utf-8")

    resolved = resolve_avatar_config_path_for_avatar_id(osc_root=osc_root, avatar_id="avtr_test.json")
    assert resolved == p

from __future__ import annotations

from pathlib import Path

from vrchat_osc_mcp.vrc_config.parser import load_avatar_schema


def test_parse_sample_schema_and_resolve():
    repo_root = Path(__file__).resolve().parents[1]
    sample_path = repo_root / "sample.json"
    assert sample_path.exists(), "sample.json should exist in repo root"

    schema = load_avatar_schema(sample_path)

    assert schema.avatar_id.startswith("avtr_")
    assert len(schema.parameters) > 0

    # Resolve a known-safe parameter if present (common patterns: ASCII alnum/underscore)
    if "BalanceBallBSProximity" in schema.parameters:
        p = schema.resolve("BalanceBallBSProximity")
        assert p is not None
        assert p.type in ("Bool", "Int", "Float")
        assert p.input_address is None or p.input_address.startswith("/avatar/parameters/")

    # Find at least one parameter whose OSC suffix differs from its display name
    mismatch = None
    for p in schema.parameters.values():
        if not p.input_address or not p.input_address.startswith("/avatar/parameters/"):
            continue
        suffix = p.input_address.removeprefix("/avatar/parameters/")
        if suffix != p.name:
            mismatch = (p, suffix)
            break

    assert mismatch is not None, "expected at least one parameter with sanitized OSC address"
    p, suffix = mismatch

    # Should resolve by suffix and full address
    assert schema.resolve(suffix) is not None
    assert schema.resolve(p.input_address) is not None


def test_parse_handles_unicode_names():
    repo_root = Path(__file__).resolve().parents[1]
    schema = load_avatar_schema(repo_root / "sample.json")

    any_unicode = next((p for p in schema.parameters.values() if any(ord(ch) > 127 for ch in p.name)), None)
    if any_unicode is None:
        return

    assert schema.resolve(any_unicode.name) is not None

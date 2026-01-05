from __future__ import annotations

from vrchat_osc_mcp.mcp_server import VRC_TOOL_NAMES, _OPENAI_TOOL_NAME_RE, create_server


def test_tool_names_are_openai_compatible() -> None:
    names = list(VRC_TOOL_NAMES.values())

    assert len(names) == len(set(names)), "Tool names must be unique"

    bad = [n for n in names if _OPENAI_TOOL_NAME_RE.fullmatch(n) is None]
    assert not bad, f"Invalid tool names: {bad!r}"


def test_create_server_registers_without_error() -> None:
    # We only care that tool registration succeeds (i.e., name validation passes).
    class _DummyAdapter:  # noqa: D401
        """Minimal adapter stub for server construction."""

    create_server(adapter=_DummyAdapter())

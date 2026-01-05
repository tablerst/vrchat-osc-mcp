from __future__ import annotations

from vrchat_osc_mcp.domain.chatbox import trim_chatbox_text


def test_chatbox_trim_chars_and_lines():
    text = "\n".join([f"line{i}" for i in range(1, 20)])
    trimmed = trim_chatbox_text(text)

    assert len(trimmed) <= 144
    assert len(trimmed.split("\n")) <= 9

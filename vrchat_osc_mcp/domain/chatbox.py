from __future__ import annotations


def trim_chatbox_text(text: str, *, max_chars: int = 144, max_lines: int = 9) -> str:
    """Trim text to VRChat chatbox limits.

    VRChat chatbox is limited to 144 chars and 9 lines. The 9-lines constraint is
    affected by auto-wrapping in-client; we enforce a strict newline-based max
    line count, then a final max-char cutoff.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if len(lines) > max_lines:
        normalized = "\n".join(lines[:max_lines])

    if len(normalized) > max_chars:
        normalized = normalized[:max_chars]

    return normalized

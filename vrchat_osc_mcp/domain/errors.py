from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DomainError(Exception):
    code: str
    message: str
    details: dict[str, Any] | None = None
    retry_after_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }
        if self.details is not None:
            payload["error"]["details"] = self.details
        if self.retry_after_ms is not None:
            payload["error"]["retry_after_ms"] = self.retry_after_ms
        return payload


def ok(data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": True, **(data or {})}

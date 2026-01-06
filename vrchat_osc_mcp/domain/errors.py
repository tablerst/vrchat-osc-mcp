from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


VRCErrorCode = Literal[
    "VRC_NOT_READY",
    "CAPABILITY_UNAVAILABLE",
    "INVALID_ARGUMENT",
    "CONFLICT",
    "STREAM_NOT_RUNNING",
    "RATE_LIMITED",
    "INTERNAL_ERROR",
]


@dataclass(frozen=True)
class DomainError(Exception):
    """Domain-level error.

    NOTE: Per v1.0 contract, the *MCP tool layer* owns the unified envelope
    (ok/data/error/trace_id). DomainError only describes the error payload.
    """

    code: VRCErrorCode
    message: str
    details: dict[str, Any] | None = None

    def to_error_obj(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            err["details"] = self.details
        return err

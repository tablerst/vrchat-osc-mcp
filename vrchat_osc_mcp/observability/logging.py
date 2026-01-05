from __future__ import annotations

import logging
from typing import Literal

import structlog

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def configure_logging(*, level: LogLevel = "INFO", json_logs: bool = True) -> None:
    """Configure stdlib logging + structlog.

    We default to JSON logs for machine parsing (MCP hosts / log collectors).
    """

    logging.basicConfig(level=getattr(logging, level), format="%(message)s")

    processors: list[structlog.types.Processor] = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "vrchat-osc-mcp") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

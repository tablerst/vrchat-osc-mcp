from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_ms: int | None


class SlidingWindowRateLimiter:
    """Non-blocking limiter: decide allow/deny and suggest retry delay."""

    def __init__(self, *, max_events: int, window_s: float) -> None:
        self._max = max_events
        self._window_s = window_s
        self._events: deque[float] = deque()

    def check(self) -> RateLimitDecision:
        now = time.monotonic()
        cutoff = now - self._window_s
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

        if len(self._events) < self._max:
            self._events.append(now)
            return RateLimitDecision(allowed=True, retry_after_ms=None)

        retry_after_s = (self._events[0] + self._window_s) - now
        retry_after_ms = max(0, int(retry_after_s * 1000))
        return RateLimitDecision(allowed=False, retry_after_ms=retry_after_ms)

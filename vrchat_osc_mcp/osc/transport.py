from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from pythonosc.udp_client import SimpleUDPClient


@dataclass(frozen=True)
class OSCOutboundMessage:
    address: str
    value: Any
    trace_id: str
    created_at: float


class AsyncSlidingWindowThrottle:
    """Async throttling: keep max N events per window seconds.

    This *delays* sends instead of erroring, which is safer for global OSC traffic.
    """

    def __init__(self, *, max_events: int, window_s: float) -> None:
        self._max = max_events
        self._window_s = window_s
        self._events: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def throttle(self) -> None:
        while True:
            sleep_s = 0.0
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self._window_s
                while self._events and self._events[0] <= cutoff:
                    self._events.popleft()

                if len(self._events) < self._max:
                    self._events.append(now)
                    return

                # Need to wait until the oldest event expires
                sleep_s = (self._events[0] + self._window_s) - now

            if sleep_s > 0:
                await asyncio.sleep(sleep_s)


class OSCTransport:
    def __init__(
        self,
        *,
        send_ip: str,
        send_port: int,
        osc_per_second: int,
        logger,
        queue_maxsize: int = 2048,
    ) -> None:
        self._client = SimpleUDPClient(send_ip, send_port)
        self._logger = logger
        self._queue: asyncio.Queue[OSCOutboundMessage] = asyncio.Queue(maxsize=queue_maxsize)
        self._task: asyncio.Task[None] | None = None
        self._throttle = AsyncSlidingWindowThrottle(max_events=osc_per_second, window_s=1.0)
        self._last_sent_at: float | None = None

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def last_sent_ms_ago(self) -> int | None:
        if self._last_sent_at is None:
            return None
        return max(0, int((time.monotonic() - self._last_sent_at) * 1000))

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="osc-sender")

    async def close(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def send(self, *, address: str, value: Any, trace_id: str) -> None:
        msg = OSCOutboundMessage(address=address, value=value, trace_id=trace_id, created_at=time.monotonic())
        await self._queue.put(msg)

    async def flush(self, *, timeout_s: float = 2.0) -> None:
        """Best-effort wait until the queue is drained."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._queue.empty():
                return
            await asyncio.sleep(0.005)

    async def _run(self) -> None:
        while True:
            msg = await self._queue.get()
            try:
                await self._throttle.throttle()
                self._client.send_message(msg.address, msg.value)
                self._last_sent_at = time.monotonic()
                self._logger.info(
                    "osc.send",
                    trace_id=msg.trace_id,
                    osc_address=msg.address,
                    osc_value=msg.value,
                )
            finally:
                self._queue.task_done()

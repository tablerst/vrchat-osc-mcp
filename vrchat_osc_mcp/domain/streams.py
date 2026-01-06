from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from .errors import DomainError

StreamState = Literal["STOPPED", "RUNNING", "STOPPING"]


@dataclass
class StreamStatus:
    running: bool
    stream_id: str | None
    fps: int | None
    mode: str | None = None


_TRACKER_TO_INDEX: dict[str, int] = {
    # VRChat 官方：最多 8 个 tracker：hip, chest, 2x feet, 2x knees, 2x elbows
    # /tracking/trackers/1..8/{position,rotation}
    "hip": 1,
    "chest": 2,
    "left_foot": 3,
    "right_foot": 4,
    "left_knee": 5,
    "right_knee": 6,
    "left_elbow": 7,
    "right_elbow": 8,
}


def tracker_to_osc_index(tracker: str) -> int:
    try:
        return _TRACKER_TO_INDEX[tracker]
    except KeyError as e:
        raise DomainError(
            code="INVALID_ARGUMENT",
            message="未知/不支持的 tracker 名称。",
            details={"given": tracker, "supported": sorted(_TRACKER_TO_INDEX.keys())},
        ) from e


def _sleep_interval_s(fps: int) -> float:
    if fps <= 0:
        return 1 / 60
    return 1.0 / float(fps)


class TrackingStream:
    def __init__(self, *, transport, logger, target_ttl_ms: int = 10_000) -> None:
        self._transport = transport
        self._logger = logger

        self._task: asyncio.Task[None] | None = None
        self._state: StreamState = "STOPPED"
        self._stream_id: str | None = None

        self._fps: int = 60
        self._enabled_trackers: list[str] = []
        self._neutral_on_stop: bool = True

        # If >0, targets older than this TTL will be neutralized and cleared.
        self._target_ttl_s: float = max(0.0, float(target_ttl_ms) / 1000.0)

        # Target pose per tracker: {tracker: {"pos": (x,y,z), "rot": (x,y,z)}}
        self._targets: dict[str, dict[str, tuple[float, float, float]]] = {}
        self._target_updated_at: dict[str, float] = {}

        # Optional head reference (used for alignment)
        self._head_pos: tuple[float, float, float] | None = None
        self._head_rot: tuple[float, float, float] | None = None
        self._head_mode: Literal["single_align", "stream_align"] = "stream_align"
        self._head_rotation_suppress_until: float = 0.0
        self._last_single_align_sent_at: float | None = None
        self._head_updated_at: float | None = None

    def status(self) -> StreamStatus:
        return StreamStatus(
            running=self._state == "RUNNING",
            stream_id=self._stream_id,
            fps=self._fps if self._state == "RUNNING" else None,
        )

    def set_tracker_pose(
        self,
        *,
        tracker: str,
        position_m: tuple[float, float, float],
        rotation_euler_deg: tuple[float, float, float],
    ) -> None:
        # Validation: tracker must be known (also guarantees mapping exists)
        tracker_to_osc_index(tracker)
        self._targets.setdefault(tracker, {})["pos"] = position_m
        self._targets.setdefault(tracker, {})["rot"] = rotation_euler_deg
        self._target_updated_at[tracker] = time.monotonic()

    async def set_head_reference(
        self,
        *,
        position_m: tuple[float, float, float] | None,
        rotation_euler_deg: tuple[float, float, float] | None,
        mode: Literal["single_align", "stream_align"],
        trace_id: str,
    ) -> None:
        self._head_pos = position_m
        self._head_rot = rotation_euler_deg
        self._head_mode = mode
        self._head_updated_at = time.monotonic()

        # Per VRChat doc: if only a single head rotation message is sent and no
        # second arrives within 300ms, it is treated as one-time instant alignment.
        if mode == "single_align" and rotation_euler_deg is not None:
            now = time.monotonic()
            if self._last_single_align_sent_at is not None:
                since = now - self._last_single_align_sent_at
                if since < 0.300:
                    await asyncio.sleep(0.300 - since)
            await self._transport.send(
                address="/tracking/trackers/head/rotation",
                value=list(rotation_euler_deg),
                trace_id=trace_id,
            )
            self._last_single_align_sent_at = time.monotonic()
            self._head_rotation_suppress_until = time.monotonic() + 0.300

    async def start(
        self,
        *,
        fps: int,
        enabled_trackers: list[str],
        neutral_on_stop: bool,
    ) -> str:
        if self._state == "RUNNING":
            # Idempotent: return existing stream_id
            assert self._stream_id is not None
            # If config differs, treat as conflict (explicit)
            if fps != self._fps or enabled_trackers != self._enabled_trackers or neutral_on_stop != self._neutral_on_stop:
                raise DomainError(
                    code="CONFLICT",
                    message="Tracking stream 已在运行，且配置不同。请先 stop 再 start。",
                    details={
                        "running": {"fps": self._fps, "enabled_trackers": self._enabled_trackers, "neutral_on_stop": self._neutral_on_stop},
                        "requested": {"fps": fps, "enabled_trackers": enabled_trackers, "neutral_on_stop": neutral_on_stop},
                    },
                )
            return self._stream_id

        self._fps = int(fps)
        self._enabled_trackers = list(enabled_trackers)
        self._neutral_on_stop = bool(neutral_on_stop)

        # Treat any pre-existing targets as fresh upon start.
        now = time.monotonic()
        for t in self._targets.keys():
            self._target_updated_at[t] = now
        if self._head_pos is not None or self._head_rot is not None:
            self._head_updated_at = now

        self._state = "RUNNING"
        self._stream_id = str(uuid.uuid4())
        self._task = asyncio.create_task(self._run(), name="vrc-tracking-stream")
        return self._stream_id

    async def stop(self, *, trace_id: str, neutral_on_stop: bool | None = None) -> None:
        if self._state == "STOPPED":
            return

        if neutral_on_stop is None:
            neutral_on_stop = self._neutral_on_stop

        self._state = "STOPPING"
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None

        if neutral_on_stop:
            await self._send_neutral(trace_id=trace_id)

        self._state = "STOPPED"
        self._stream_id = None

    async def _send_neutral(self, *, trace_id: str) -> None:
        # Best-effort: send a single neutral frame
        for tracker in self._enabled_trackers:
            idx = tracker_to_osc_index(tracker)
            await self._transport.send(address=f"/tracking/trackers/{idx}/position", value=[0.0, 0.0, 0.0], trace_id=trace_id)
            await self._transport.send(address=f"/tracking/trackers/{idx}/rotation", value=[0.0, 0.0, 0.0], trace_id=trace_id)

        # Also neutral head, if used
        await self._transport.send(address="/tracking/trackers/head/position", value=[0.0, 0.0, 0.0], trace_id=trace_id)
        await self._transport.send(address="/tracking/trackers/head/rotation", value=[0.0, 0.0, 0.0], trace_id=trace_id)

    async def _run(self) -> None:
        interval = _sleep_interval_s(self._fps)
        while True:
            now = time.monotonic()

            # Trackers
            for tracker in self._enabled_trackers:
                idx = tracker_to_osc_index(tracker)
                t = self._targets.get(tracker)
                if not t:
                    continue

                # Safety: auto-neutralize stale targets.
                if self._target_ttl_s > 0:
                    updated_at = self._target_updated_at.get(tracker)
                    if updated_at is None or (now - updated_at) > self._target_ttl_s:
                        await self._transport.send(
                            address=f"/tracking/trackers/{idx}/position",
                            value=[0.0, 0.0, 0.0],
                            trace_id=self._stream_id or "",
                        )
                        await self._transport.send(
                            address=f"/tracking/trackers/{idx}/rotation",
                            value=[0.0, 0.0, 0.0],
                            trace_id=self._stream_id or "",
                        )
                        self._targets.pop(tracker, None)
                        self._target_updated_at.pop(tracker, None)
                        continue

                if (pos := t.get("pos")) is not None:
                    await self._transport.send(address=f"/tracking/trackers/{idx}/position", value=list(pos), trace_id=self._stream_id or "")
                if (rot := t.get("rot")) is not None:
                    await self._transport.send(address=f"/tracking/trackers/{idx}/rotation", value=list(rot), trace_id=self._stream_id or "")

            # Head reference
            if (self._head_pos is not None or self._head_rot is not None) and self._target_ttl_s > 0:
                if self._head_updated_at is None or (now - self._head_updated_at) > self._target_ttl_s:
                    # Auto-neutralize stale head reference.
                    await self._transport.send(
                        address="/tracking/trackers/head/position",
                        value=[0.0, 0.0, 0.0],
                        trace_id=self._stream_id or "",
                    )
                    await self._transport.send(
                        address="/tracking/trackers/head/rotation",
                        value=[0.0, 0.0, 0.0],
                        trace_id=self._stream_id or "",
                    )
                    self._head_pos = None
                    self._head_rot = None
                    self._head_updated_at = None

            if self._head_pos is not None:
                await self._transport.send(address="/tracking/trackers/head/position", value=list(self._head_pos), trace_id=self._stream_id or "")

            if self._head_rot is not None and self._head_mode == "stream_align" and now >= self._head_rotation_suppress_until:
                await self._transport.send(address="/tracking/trackers/head/rotation", value=list(self._head_rot), trace_id=self._stream_id or "")

            await asyncio.sleep(interval)


_GAZE_MODES = {
    "CenterPitchYaw",
    "CenterPitchYawDist",
    "CenterVec",
    "CenterVecFull",
    "LeftRightPitchYaw",
    "LeftRightVec",
}


def validate_gaze_mode(mode: str) -> str:
    if mode not in _GAZE_MODES:
        raise DomainError(
            code="INVALID_ARGUMENT",
            message="未知/不支持的 gaze_mode。",
            details={"given": mode, "supported": sorted(_GAZE_MODES)},
        )
    return mode


def _neutral_gaze_args(mode: str) -> list[float]:
    # Neutral look forward.
    if mode == "CenterPitchYaw":
        return [0.0, 0.0]
    if mode == "CenterPitchYawDist":
        return [0.0, 0.0, 1.0]
    if mode == "CenterVec":
        return [0.0, 0.0, 1.0]
    if mode == "CenterVecFull":
        return [0.0, 0.0, 1.0]
    if mode == "LeftRightPitchYaw":
        return [0.0, 0.0, 0.0, 0.0]
    if mode == "LeftRightVec":
        return [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    return [0.0, 0.0]


class EyeStream:
    def __init__(self, *, transport, logger, target_ttl_ms: int = 10_000) -> None:
        self._transport = transport
        self._logger = logger

        self._task: asyncio.Task[None] | None = None
        self._state: StreamState = "STOPPED"
        self._stream_id: str | None = None

        self._fps: int = 60
        self._gaze_mode: str = "CenterPitchYaw"
        self._neutral_on_stop: bool = True

        # If >0, blink/gaze targets older than this TTL will revert to neutral.
        self._target_ttl_s: float = max(0.0, float(target_ttl_ms) / 1000.0)

        self._blink_amount: float = 0.0
        self._gaze_args: list[float] = _neutral_gaze_args("CenterPitchYaw")

        self._blink_updated_at: float | None = None
        self._gaze_updated_at: float | None = None

    def status(self) -> StreamStatus:
        return StreamStatus(
            running=self._state == "RUNNING",
            stream_id=self._stream_id,
            fps=self._fps if self._state == "RUNNING" else None,
            mode=self._gaze_mode if self._state == "RUNNING" else None,
        )

    def current_gaze_mode(self) -> str:
        return self._gaze_mode

    def set_blink(self, *, amount: float) -> None:
        if not (0.0 <= amount <= 1.0):
            raise DomainError(code="INVALID_ARGUMENT", message="blink amount 必须在 [0,1]。", details={"amount": amount})
        self._blink_amount = float(amount)
        self._blink_updated_at = time.monotonic()

    def set_gaze(self, *, gaze_mode: str, args: list[float]) -> None:
        gaze_mode = validate_gaze_mode(gaze_mode)
        # If running, mode is locked
        if self._state == "RUNNING" and gaze_mode != self._gaze_mode:
            raise DomainError(
                code="CONFLICT",
                message="Eye stream 运行中不允许切换 gaze_mode。请先 stop 再 start。",
                details={"running_mode": self._gaze_mode, "requested_mode": gaze_mode},
            )

        self._gaze_mode = gaze_mode
        self._gaze_args = list(args)
        self._gaze_updated_at = time.monotonic()

    async def start(self, *, fps: int, gaze_mode: str, neutral_on_stop: bool) -> str:
        gaze_mode = validate_gaze_mode(gaze_mode)

        if self._state == "RUNNING":
            assert self._stream_id is not None
            if fps != self._fps or gaze_mode != self._gaze_mode or neutral_on_stop != self._neutral_on_stop:
                raise DomainError(
                    code="CONFLICT",
                    message="Eye stream 已在运行，且配置不同。请先 stop 再 start。",
                    details={
                        "running": {"fps": self._fps, "gaze_mode": self._gaze_mode, "neutral_on_stop": self._neutral_on_stop},
                        "requested": {"fps": fps, "gaze_mode": gaze_mode, "neutral_on_stop": neutral_on_stop},
                    },
                )
            return self._stream_id

        self._fps = int(fps)
        self._gaze_mode = gaze_mode
        self._neutral_on_stop = bool(neutral_on_stop)

        # Treat pre-existing targets as fresh upon start.
        now = time.monotonic()
        if self._blink_updated_at is not None:
            self._blink_updated_at = now
        if self._gaze_updated_at is not None:
            self._gaze_updated_at = now

        # Ensure we have a gaze payload matching the configured mode
        if self._gaze_args is None or len(self._gaze_args) == 0:
            self._gaze_args = _neutral_gaze_args(self._gaze_mode)

        self._state = "RUNNING"
        self._stream_id = str(uuid.uuid4())
        self._task = asyncio.create_task(self._run(), name="vrc-eye-stream")
        return self._stream_id

    async def stop(self, *, trace_id: str, neutral_on_stop: bool | None = None) -> None:
        if self._state == "STOPPED":
            return

        if neutral_on_stop is None:
            neutral_on_stop = self._neutral_on_stop

        self._state = "STOPPING"
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None

        if neutral_on_stop:
            await self._send_neutral(trace_id=trace_id)

        self._state = "STOPPED"
        self._stream_id = None

    async def _send_neutral(self, *, trace_id: str) -> None:
        await self._transport.send(address="/tracking/eye/EyesClosedAmount", value=float(0.0), trace_id=trace_id)
        await self._transport.send(address=f"/tracking/eye/{self._gaze_mode}", value=_neutral_gaze_args(self._gaze_mode), trace_id=trace_id)

    async def _run(self) -> None:
        interval = _sleep_interval_s(self._fps)
        while True:
            now = time.monotonic()

            # Safety: auto-revert stale targets to neutral even if the stream keeps running.
            if self._target_ttl_s > 0:
                if self._blink_updated_at is not None and (now - self._blink_updated_at) > self._target_ttl_s:
                    self._blink_amount = 0.0
                    self._blink_updated_at = None
                if self._gaze_updated_at is not None and (now - self._gaze_updated_at) > self._target_ttl_s:
                    self._gaze_args = _neutral_gaze_args(self._gaze_mode)
                    self._gaze_updated_at = None

            # Eyelids keepalive (timeout is separate)
            await self._transport.send(
                address="/tracking/eye/EyesClosedAmount",
                value=float(self._blink_amount),
                trace_id=self._stream_id or "",
            )

            # Eye-look keepalive (only one address format at a time)
            await self._transport.send(
                address=f"/tracking/eye/{self._gaze_mode}",
                value=list(self._gaze_args) if self._gaze_args else _neutral_gaze_args(self._gaze_mode),
                trace_id=self._stream_id or "",
            )

            await asyncio.sleep(interval)

from __future__ import annotations

import asyncio
import inspect
import re
import uuid
from typing import Any

from fastmcp import Context, FastMCP

from .domain.errors import DomainError
from .observability.logging import get_logger

# OpenAI function/tool name constraints (used by some MCP clients):
# - Allowed chars: A-Z a-z 0-9 _ -
# - Length capped (historically 64)
_OPENAI_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

VRC_V1_TOOL_NAMES: tuple[str, ...] = (
    # Meta
    "vrc_meta_get_status",
    "vrc_meta_get_capabilities",
    # Input
    "vrc_input_list_endpoints",
    "vrc_input_set_axes",
    "vrc_input_tap_buttons",
    "vrc_input_hold_buttons",
    "vrc_input_release_buttons",
    "vrc_input_stop",
    # Avatar
    "vrc_avatar_list_parameters",
    "vrc_avatar_set_parameter",
    "vrc_avatar_set_parameters",
    # Tracking
    "vrc_tracking_set_tracker_pose",
    "vrc_tracking_set_head_reference",
    "vrc_tracking_stream_start",
    "vrc_tracking_stream_stop",
    # Eye
    "vrc_eye_stream_start",
    "vrc_eye_set_blink",
    "vrc_eye_set_gaze",
    "vrc_eye_stream_stop",
    # Chatbox
    "vrc_chatbox_send",
    "vrc_chatbox_set_typing",
    # Global safety
    "vrc_stop_all",
    # Macro
    "vrc_macro_move_for",
    "vrc_macro_turn_degrees",
    "vrc_macro_look_at",
    "vrc_macro_emote",
    "vrc_macro_idle",
    "vrc_macro_stop",
)


def create_server(*, adapter) -> FastMCP:
    mcp = FastMCP(
        name="vrchat-osc-mcp",
        # Avoid leaking internals by default; our DomainError payload is explicit anyway.
        mask_error_details=True,
    )

    logger = get_logger().bind(component="mcp")

    def _trace_id(ctx: Context | None) -> str:
        rid = getattr(ctx, "request_id", None)
        return rid or str(uuid.uuid4())

    def _ok(*, trace_id: str, data: dict | list | str | int | float | bool | None) -> dict:
        return {"ok": True, "data": data, "error": None, "trace_id": trace_id}

    def _err(*, trace_id: str, error_obj: dict) -> dict:
        return {"ok": False, "data": None, "error": error_obj, "trace_id": trace_id}

    def _wrap_sync(fn):
        def _inner(*args, **kwargs):
            ctx = kwargs.pop("ctx", None)
            trace_id = _trace_id(ctx) if isinstance(ctx, Context) else _trace_id(None)
            try:
                data = fn(*args, **kwargs, trace_id=trace_id)
                return _ok(trace_id=trace_id, data=data)
            except DomainError as e:
                return _err(trace_id=trace_id, error_obj=e.to_error_obj())
            except Exception as e:  # noqa: BLE001
                logger.exception("tool.internal_error", trace_id=trace_id, error=str(e))
                return _err(
                    trace_id=trace_id,
                    error_obj={"code": "INTERNAL_ERROR", "message": "Internal error"},
                )

        return _inner

    def _wrap_async(fn):
        async def _inner(*args, **kwargs):
            ctx = kwargs.pop("ctx", None)
            trace_id = _trace_id(ctx) if isinstance(ctx, Context) else _trace_id(None)
            try:
                data = await fn(*args, **kwargs, trace_id=trace_id)
                return _ok(trace_id=trace_id, data=data)
            except DomainError as e:
                return _err(trace_id=trace_id, error_obj=e.to_error_obj())
            except Exception as e:  # noqa: BLE001
                logger.exception("tool.internal_error", trace_id=trace_id, error=str(e))
                return _err(
                    trace_id=trace_id,
                    error_obj={"code": "INTERNAL_ERROR", "message": "Internal error"},
                )

        return _inner

    _ENVELOPE_OUTPUT_SCHEMA: dict = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "data": {},
            "error": {"type": ["object", "null"]},
            "trace_id": {"type": "string"},
        },
        "required": ["ok", "data", "error", "trace_id"],
        "additionalProperties": False,
    }

    _STATUS_OUTPUT_SCHEMA: dict = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "data": {
                "type": ["object", "null"],
                "properties": {
                    "osc_enabled_detected": {"type": "boolean"},
                    "target_host": {"type": ["string", "null"]},
                    "target_port": {"type": ["integer", "null"]},
                    "last_send_ms_ago": {"type": ["integer", "null"]},
                    "streams": {
                        "type": "object",
                        "properties": {
                            "tracking": {
                                "type": "object",
                                "properties": {
                                    "running": {"type": "boolean"},
                                    "stream_id": {"type": ["string", "null"]},
                                    "fps": {"type": ["integer", "null"]},
                                },
                                "additionalProperties": False,
                            },
                            "eye": {
                                "type": "object",
                                "properties": {
                                    "running": {"type": "boolean"},
                                    "stream_id": {"type": ["string", "null"]},
                                    "fps": {"type": ["integer", "null"]},
                                    "mode": {"type": ["string", "null"]},
                                },
                                "additionalProperties": False,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
            "error": {"type": ["object", "null"]},
            "trace_id": {"type": "string"},
        },
        "required": ["ok", "data", "error", "trace_id"],
        "additionalProperties": False,
    }

    def _tool(*, name: str, **kwargs):
        # Fail fast if someone adds an incompatible name.
        if not _OPENAI_TOOL_NAME_RE.fullmatch(name):
            raise ValueError(
                f"Invalid MCP tool name: {name!r}. "
                "Tool names must match ^[A-Za-z0-9_-]{1,64}$."
            )

        def _decorator(fn):
            # Do not expose MCP Context injection params to clients.
            # (Per user contract: trace_id is output-only; ctx should not appear in inputSchema.)
            tool_kwargs = dict(kwargs)
            try:
                sig = inspect.signature(fn)
                if "ctx" in sig.parameters:
                    tool_kwargs.setdefault("exclude_args", ["ctx"])
            except Exception:
                # If we cannot inspect, just proceed without exclude_args.
                pass

            # Back-compat: older fastmcp versions may not support exclude_args.
            try:
                return mcp.tool(name=name, **tool_kwargs)(fn)
            except TypeError:
                tool_kwargs.pop("exclude_args", None)
                return mcp.tool(name=name, **tool_kwargs)(fn)

        return _decorator

    # -----------------
    # Meta
    # -----------------

    @_tool(name="vrc_meta_get_status", annotations={"readOnlyHint": True}, output_schema=_STATUS_OUTPUT_SCHEMA)
    def vrc_meta_get_status() -> dict:
        return _wrap_sync(lambda *, trace_id: adapter.meta_get_status())()

    @_tool(name="vrc_meta_get_capabilities", annotations={"readOnlyHint": True}, output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    def vrc_meta_get_capabilities(refresh: bool = False) -> dict:
        return _wrap_sync(lambda refresh, *, trace_id: adapter.meta_get_capabilities(refresh=refresh))(refresh)

    # -----------------
    # Input
    # -----------------

    @_tool(name="vrc_input_list_endpoints", annotations={"readOnlyHint": True}, output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    def vrc_input_list_endpoints() -> dict:
        return _wrap_sync(lambda *, trace_id: adapter.input_list_endpoints())()

    @_tool(name="vrc_input_set_axes", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_input_set_axes(
        axes: dict,
        duration_ms: int = 0,
        auto_zero: bool = True,
        ease_ms: int = 80,
        ctx: Context | None = None,
    ) -> dict:
        return await _wrap_async(adapter.input_set_axes)(
            axes=axes,
            duration_ms=duration_ms,
            auto_zero=auto_zero,
            ease_ms=ease_ms,
            ctx=ctx,
        )

    @_tool(name="vrc_input_tap_buttons", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_input_tap_buttons(buttons: list[str], press_ms: int = 80, ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.input_tap_buttons)(buttons=buttons, press_ms=press_ms, ctx=ctx)

    @_tool(name="vrc_input_hold_buttons", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_input_hold_buttons(buttons: list[str], ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.input_hold_buttons)(buttons=buttons, ctx=ctx)

    @_tool(name="vrc_input_release_buttons", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_input_release_buttons(buttons: list[str], ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.input_release_buttons)(buttons=buttons, ctx=ctx)

    @_tool(name="vrc_input_stop", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_input_stop(ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.input_stop)(ctx=ctx)

    # -----------------
    # Avatar
    # -----------------

    @_tool(name="vrc_avatar_list_parameters", annotations={"readOnlyHint": True}, output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    def vrc_avatar_list_parameters() -> dict:
        return _wrap_sync(lambda *, trace_id: adapter.avatar_list_parameters())()

    @_tool(name="vrc_avatar_set_parameter", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_avatar_set_parameter(
        name: str,
        value: Any,
        duration_ms: int = 0,
        reset_value: Any = None,
        ctx: Context | None = None,
    ) -> dict:
        return await _wrap_async(adapter.avatar_set_parameter)(
            name=name,
            value=value,
            duration_ms=duration_ms,
            reset_value=reset_value,
            ctx=ctx,
        )

    @_tool(name="vrc_avatar_set_parameters", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_avatar_set_parameters(parameters: list[dict], ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.avatar_set_parameters)(parameters=parameters, ctx=ctx)

    # -----------------
    # Tracking
    # -----------------

    @_tool(name="vrc_tracking_set_tracker_pose", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_tracking_set_tracker_pose(
        tracker: str,
        position_m: dict,
        rotation_euler_deg: dict,
        blend_ms: int = 60,
        space: str = "unity",
        ctx: Context | None = None,
    ) -> dict:
        # v1: space only supports "unity" (per PLAN)
        if space != "unity":
            trace_id = _trace_id(ctx)
            return _err(
                trace_id=trace_id,
                error_obj={"code": "INVALID_ARGUMENT", "message": "space 仅支持 unity。", "details": {"space": space}},
            )
        if blend_ms < 0 or blend_ms > 1000:
            trace_id = _trace_id(ctx)
            return _err(
                trace_id=trace_id,
                error_obj={"code": "INVALID_ARGUMENT", "message": "blend_ms 必须在 [0,1000]。", "details": {"blend_ms": blend_ms}},
            )
        # blend_ms currently accepted but not applied (reserved)
        return await _wrap_async(adapter.tracking_set_tracker_pose)(
            tracker=tracker,
            position_m=position_m,
            rotation_euler_deg=rotation_euler_deg,
            ctx=ctx,
        )

    @_tool(name="vrc_tracking_set_head_reference", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_tracking_set_head_reference(
        position_m: dict | None = None,
        rotation_euler_deg: dict | None = None,
        mode: str = "stream_align",
        ctx: Context | None = None,
    ) -> dict:
        return await _wrap_async(adapter.tracking_set_head_reference)(
            position_m=position_m,
            rotation_euler_deg=rotation_euler_deg,
            mode=mode,
            ctx=ctx,
        )

    @_tool(name="vrc_tracking_stream_start", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_tracking_stream_start(
        fps: int = 60,
        enabled_trackers: list[str] | None = None,
        neutral_on_stop: bool = True,
        ctx: Context | None = None,
    ) -> dict:
        if fps not in (20, 30, 45, 60):
            trace_id = _trace_id(ctx)
            return _err(
                trace_id=trace_id,
                error_obj={"code": "INVALID_ARGUMENT", "message": "fps 必须是 20/30/45/60。", "details": {"fps": fps}},
            )
        if enabled_trackers is None:
            enabled_trackers = ["hip", "left_foot", "right_foot"]
        return await _wrap_async(adapter.tracking_stream_start)(
            fps=fps,
            enabled_trackers=enabled_trackers,
            neutral_on_stop=neutral_on_stop,
            ctx=ctx,
        )

    @_tool(name="vrc_tracking_stream_stop", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_tracking_stream_stop(stream_id: str | None = None, ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.tracking_stream_stop)(stream_id=stream_id, ctx=ctx)

    # -----------------
    # Eye
    # -----------------

    @_tool(name="vrc_eye_stream_start", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_eye_stream_start(
        fps: int = 60,
        gaze_mode: str = "CenterPitchYaw",
        neutral_on_stop: bool = True,
        ctx: Context | None = None,
    ) -> dict:
        if fps not in (20, 30, 45, 60):
            trace_id = _trace_id(ctx)
            return _err(
                trace_id=trace_id,
                error_obj={"code": "INVALID_ARGUMENT", "message": "fps 必须是 20/30/45/60。", "details": {"fps": fps}},
            )
        return await _wrap_async(adapter.eye_stream_start)(fps=fps, gaze_mode=gaze_mode, neutral_on_stop=neutral_on_stop, ctx=ctx)

    @_tool(name="vrc_eye_set_blink", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_eye_set_blink(amount: float, ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.eye_set_blink)(amount=amount, ctx=ctx)

    @_tool(name="vrc_eye_set_gaze", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_eye_set_gaze(gaze_mode: str, data: dict, ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.eye_set_gaze)(gaze_mode=gaze_mode, data=data, ctx=ctx)

    @_tool(name="vrc_eye_stream_stop", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_eye_stream_stop(stream_id: str | None = None, ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.eye_stream_stop)(stream_id=stream_id, ctx=ctx)

    # -----------------
    # Chatbox
    # -----------------

    @_tool(name="vrc_chatbox_send", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_chatbox_send(
        text: str,
        send_immediately: bool = True,
        notify: bool = True,
        set_typing: bool = False,
        ctx: Context | None = None,
    ) -> dict:
        return await _wrap_async(adapter.chatbox_send)(
            text=text,
            send_immediately=send_immediately,
            notify=notify,
            set_typing=set_typing,
            ctx=ctx,
        )

    @_tool(name="vrc_chatbox_set_typing", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_chatbox_set_typing(is_typing: bool, ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.chatbox_set_typing)(is_typing=is_typing, ctx=ctx)

    # -----------------
    # Global safety
    # -----------------

    @_tool(name="vrc_stop_all", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_stop_all(ctx: Context | None = None) -> dict:
        return await _wrap_async(adapter.stop_all)(ctx=ctx)

    # -----------------
    # Macro
    # -----------------

    @_tool(name="vrc_macro_move_for", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_macro_move_for(
        duration_ms: int,
        forward: float = 0,
        strafe: float = 0,
        turn: float = 0,
        ctx: Context | None = None,
    ) -> dict:
        async def _impl(*, trace_id: str):
            if duration_ms < 50 or duration_ms > 30000:
                raise DomainError(code="INVALID_ARGUMENT", message="duration_ms 必须在 [50,30000]。", details={"duration_ms": duration_ms})
            for name, v in ("forward", forward), ("strafe", strafe), ("turn", turn):
                if v < -1 or v > 1:
                    raise DomainError(code="INVALID_ARGUMENT", message=f"{name} 必须在 [-1,1]。", details={name: v})
            return await adapter.input_set_axes(
                axes={"Vertical": forward, "Horizontal": strafe, "LookHorizontal": turn},
                duration_ms=duration_ms,
                auto_zero=True,
                ease_ms=80,
                trace_id=trace_id,
            )

        return await _wrap_async(_impl)(ctx=ctx)

    @_tool(name="vrc_macro_turn_degrees", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_macro_turn_degrees(degrees: float, ctx: Context | None = None) -> dict:
        # Simple heuristic: map degrees -> LookHorizontal axis pulse duration.
        # This is intentionally conservative; platform differences are noted in capabilities.
        async def _impl(*, trace_id: str):
            deg = float(degrees)
            if deg == 0:
                return {"turned_degrees": 0.0}
            sign = 1.0 if deg > 0 else -1.0
            dur = int(min(3000, max(50, abs(deg) * 10)))
            await adapter.input_set_axes(
                axes={"LookHorizontal": sign},
                duration_ms=dur,
                auto_zero=True,
                ease_ms=80,
                trace_id=trace_id,
            )
            return {"requested_degrees": deg, "used_duration_ms": dur}

        return await _wrap_async(_impl)(ctx=ctx)

    @_tool(name="vrc_macro_look_at", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_macro_look_at(
        yaw_deg: float,
        pitch_deg: float,
        duration_ms: int = 800,
        prefer_eye: bool = True,
        ctx: Context | None = None,
    ) -> dict:
        async def _impl(*, trace_id: str):
            if yaw_deg < -90 or yaw_deg > 90:
                raise DomainError(code="INVALID_ARGUMENT", message="yaw_deg 必须在 [-90,90]。", details={"yaw_deg": yaw_deg})
            if pitch_deg < -45 or pitch_deg > 45:
                raise DomainError(code="INVALID_ARGUMENT", message="pitch_deg 必须在 [-45,45]。", details={"pitch_deg": pitch_deg})
            if duration_ms < 50 or duration_ms > 10000:
                raise DomainError(code="INVALID_ARGUMENT", message="duration_ms 必须在 [50,10000]。", details={"duration_ms": duration_ms})
            if prefer_eye:
                # Ensure eye stream is running (idempotent if already started).
                try:
                    await adapter.eye_stream_start(fps=60, gaze_mode="CenterPitchYaw", neutral_on_stop=True, trace_id=trace_id)
                except DomainError:
                    # Fall back to input-based turning
                    pass
                else:
                    await adapter.eye_set_gaze(
                        gaze_mode="CenterPitchYaw",
                        data={"pitch": float(pitch_deg), "yaw": float(yaw_deg)},
                        trace_id=trace_id,
                    )
                    await asyncio.sleep(max(0, int(duration_ms)) / 1000)
                    await adapter.eye_set_gaze(
                        gaze_mode="CenterPitchYaw",
                        data={"pitch": 0.0, "yaw": 0.0},
                        trace_id=trace_id,
                    )
                    return {"mode": "eye", "yaw_deg": float(yaw_deg), "pitch_deg": float(pitch_deg)}

            # Fallback: small turn using LookHorizontal
            dur = int(min(1500, max(50, abs(float(yaw_deg)) * 10)))
            sign = 1.0 if float(yaw_deg) > 0 else -1.0
            await adapter.input_set_axes(
                axes={"LookHorizontal": sign},
                duration_ms=dur,
                auto_zero=True,
                ease_ms=80,
                trace_id=trace_id,
            )
            return {"mode": "input", "used_duration_ms": dur}

        return await _wrap_async(_impl)(ctx=ctx)

    @_tool(name="vrc_macro_emote", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    def vrc_macro_emote(name: str, ctx: Context | None = None) -> dict:
        # Placeholder: v1 需要 profile/emote 映射表。
        trace_id = _trace_id(ctx)
        return _err(
            trace_id=trace_id,
            error_obj={
                "code": "CAPABILITY_UNAVAILABLE",
                "message": "macro_emote 尚未配置（需要 emote 名 -> avatar 参数/动作 的映射表）。",
                "details": {"name": name},
            },
        )

    @_tool(name="vrc_macro_idle", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_macro_idle(ctx: Context | None = None) -> dict:
        async def _impl(*, trace_id: str):
            await adapter.input_stop(trace_id=trace_id)
            await adapter.chatbox_set_typing(is_typing=False, trace_id=trace_id)
            return {"idle": True}

        return await _wrap_async(_impl)(ctx=ctx)

    @_tool(name="vrc_macro_stop", output_schema=_ENVELOPE_OUTPUT_SCHEMA)
    async def vrc_macro_stop(ctx: Context | None = None) -> dict:
        # Per PLAN: macro_stop 直接调用 stop_all
        return await _wrap_async(adapter.stop_all)(ctx=ctx)

    return mcp

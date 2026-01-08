import asyncio
import inspect
import re
import uuid
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

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

    @_tool(
        name="vrc_meta_get_status",
        description="Get service status (OSC detection, target address, last send time, tracking/eye stream status).",
        annotations={"readOnlyHint": True},
        output_schema=_STATUS_OUTPUT_SCHEMA,
    )
    def vrc_meta_get_status() -> dict:
        """Query vrchat-osc-mcp current status.

        Purpose: Allows callers to determine if VRChat/OSC appears available and whether background tracking/eye streams are running.
        """
        return _wrap_sync(lambda *, trace_id: adapter.meta_get_status())()

    @_tool(
        name="vrc_meta_get_capabilities",
        description=(
            "Get a capability manifest (supported /input axes/buttons, tracking/eye/chatbox availability, avatar schema confidence, etc.)."
        ),
        annotations={"readOnlyHint": True},
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    def vrc_meta_get_capabilities(
        refresh: Annotated[
            bool,
            Field(
                description=(
                    "Whether to refresh capabilities/cache (v1 reserved field; current implementation is best-effort)."
                ),
            ),
        ] = False,
    ) -> dict:
        """Get current service capabilities and hints.

        Recommendation: read capabilities before taking actions to avoid the LLM "randomly pressing buttons".
        """
        return _wrap_sync(lambda refresh, *, trace_id: adapter.meta_get_capabilities(refresh=refresh))(refresh)

    # -----------------
    # Input
    # -----------------

    @_tool(
        name="vrc_input_list_endpoints",
        description=(
            "List the /input axis and button names supported by this server (v1). Call this first to discover valid names."
        ),
        annotations={"readOnlyHint": True},
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    def vrc_input_list_endpoints() -> dict:
        """List supported /input endpoints (axes/buttons)."""
        return _wrap_sync(lambda *, trace_id: adapter.input_list_endpoints())()

    @_tool(
        name="vrc_input_set_axes",
        description=(
            "Set /input axes (e.g., move/turn). Values are clamped to [-1,1]. "
            "Regardless of auto_zero, the server enforces a bounded return-to-zero to avoid stuck inputs."
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_input_set_axes(
        axes: Annotated[
            dict,
            Field(
                description=(
                    "Axis values to set. Keys are axis names (v1: Vertical/Horizontal/LookHorizontal), values are numbers. "
                    "Values outside [-1,1] are clamped; unknown axis names raise an error."
                ),
                examples=[{"Vertical": 1.0}],
            ),
        ],
        duration_ms: Annotated[
            int,
            Field(
                description=(
                    "Duration (ms). If <= 0, ease_ms is used as the default duration; the final duration is limited to safety.max_axis_duration_ms. "
                    "Note: for safety the tool enforces auto-zero (enforced_auto_zero)."
                ),
                ge=0,
            ),
        ] = 0,
        auto_zero: Annotated[
            bool,
            Field(
                description=(
                    "Whether to request auto-zero (requested_auto_zero). Even if false, the server still enforces a bounded return-to-zero to avoid "
                    "perpetual motion-style movement."
                )
            ),
        ] = True,
        ease_ms: Annotated[
            int,
            Field(
                description=(
                    "Easing/interpolation duration (ms). Used as the default duration when duration_ms<=0; range [0,500]."
                ),
                ge=0,
                le=500,
            ),
        ] = 80,
        ctx: Context | None = None,
    ) -> dict:
        """Set one or more /input axes.

        Key semantics (safety valves):
        - Axis values are clamped to [-1,1]
        - The final duration is limited to settings.safety.max_axis_duration_ms
        - Auto-zero is enforced regardless of auto_zero (to avoid stuck inputs)
        """
        return await _wrap_async(adapter.input_set_axes)(
            axes=axes,
            duration_ms=duration_ms,
            auto_zero=auto_zero,
            ease_ms=ease_ms,
            ctx=ctx,
        )

    @_tool(
        name="vrc_input_tap_buttons",
        description=(
            "Tap buttons (automatic press + release). For transient actions like Jump/Use; avoids long-press side effects."
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_input_tap_buttons(
        buttons: Annotated[
            list[str],
            Field(
                description="Button names to tap (call vrc_input_list_endpoints first). Up to 10.",
                min_length=1,
                max_length=10,
            ),
        ],
        press_ms: Annotated[
            int,
            Field(
                description="Press duration (ms). Clamped to [20,1000] to improve VRChat recognition reliability.",
                ge=0,
                le=1000,
            ),
        ] = 80,
        ctx: Context | None = None,
    ) -> dict:
        """Tap one or more /input buttons.

        Semantics: for each button, send 1 and then guarantee sending 0 in a finally block (even if an exception occurs).
        """
        return await _wrap_async(adapter.input_tap_buttons)(buttons=buttons, press_ms=press_ms, ctx=ctx)

    @_tool(
        name="vrc_input_hold_buttons",
        description=(
            "Hold buttons (send value=1 and keep held). Call vrc_input_release_buttons or vrc_input_stop later to release."
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_input_hold_buttons(
        buttons: Annotated[
            list[str],
            Field(
                description="Button names to hold (up to 10).",
                min_length=1,
                max_length=10,
            ),
        ],
        ctx: Context | None = None,
    ) -> dict:
        """Hold buttons."""
        return await _wrap_async(adapter.input_hold_buttons)(buttons=buttons, ctx=ctx)

    @_tool(
        name="vrc_input_release_buttons",
        description="Release buttons (send value=0).",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_input_release_buttons(
        buttons: Annotated[
            list[str],
            Field(
                description="Button names to release (up to 10).",
                min_length=1,
                max_length=10,
            ),
        ],
        ctx: Context | None = None,
    ) -> dict:
        """Release buttons."""
        return await _wrap_async(adapter.input_release_buttons)(buttons=buttons, ctx=ctx)

    @_tool(
        name="vrc_input_stop",
        description="Emergency stop for input: zero all axes and release all buttons (idempotent).",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_input_stop(ctx: Context | None = None) -> dict:
        """Stop all /input effects (zero + release)."""
        return await _wrap_async(adapter.input_stop)(ctx=ctx)

    # -----------------
    # Avatar
    # -----------------

    @_tool(
        name="vrc_avatar_list_parameters",
        description=(
            "List current Avatar parameters (from local OSC config/schema; may be empty if not loaded)."
        ),
        annotations={"readOnlyHint": True},
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    def vrc_avatar_list_parameters() -> dict:
        """List the current avatar's parameters (returns empty if schema is not loaded)."""
        return _wrap_sync(lambda *, trace_id: adapter.avatar_list_parameters())()

    @_tool(
        name="vrc_avatar_set_parameter",
        description=(
            "Set a single Avatar parameter. Supports duration_ms + reset_value pulses; parameter policy is controlled by safety.parameter_policy."
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_avatar_set_parameter(
        name: Annotated[
            str,
            Field(
                description=(
                    "Parameter name (recommended: call vrc_avatar_list_parameters first; you may also pass a raw VRChat parameter name)."
                ),
                min_length=1,
            ),
        ],
        value: Any,
        duration_ms: Annotated[
            int,
            Field(
                description=(
                    "Optional: duration (ms). If >0, after waiting it automatically writes back reset_value/default."
                ),
                ge=0,
            ),
        ] = 0,
        reset_value: Any = None,
        ctx: Context | None = None,
    ) -> dict:
        """Set a single Avatar parameter.

        Notes:
        - Best practice for value / reset_value is bool/int/float (VRChat parameter types).
        - Under the strict policy, the schema is validated for existence and parameter writability.
        """
        return await _wrap_async(adapter.avatar_set_parameter)(
            name=name,
            value=value,
            duration_ms=duration_ms,
            reset_value=reset_value,
            ctx=ctx,
        )

    @_tool(
        name="vrc_avatar_set_parameters",
        description="Set multiple Avatar parameters in one call (reduces tool calls).",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_avatar_set_parameters(
        parameters: Annotated[
            list[dict],
            Field(
                description="Array of parameter objects: [{name, value}, ...]. At least 1; recommended maximum is 64.",
                min_length=1,
            ),
        ],
        ctx: Context | None = None,
    ) -> dict:
        """Set parameters in bulk (sent one-by-one internally; returns per-item results)."""
        return await _wrap_async(adapter.avatar_set_parameters)(parameters=parameters, ctx=ctx)

    # -----------------
    # Tracking
    # -----------------

    @_tool(
        name="vrc_tracking_set_tracker_pose",
        description=(
            "Set a tracker's target pose (position_m / rotation_euler_deg). Typically used together with the tracking stream."
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_tracking_set_tracker_pose(
        tracker: Annotated[
            str,
            Field(
                description=(
                    "Tracker name (e.g., hip/left_foot/right_foot). "
                    "Use vrc_meta_get_capabilities to see recommended trackers."
                ),
                min_length=1,
            ),
        ],
        position_m: Annotated[
            dict,
            Field(description="Position in meters. Object: {x,y,z}. Coordinate system assumes Unity."),
        ],
        rotation_euler_deg: Annotated[
            dict,
            Field(description="Euler rotation in degrees. Object: {x,y,z}."),
        ],
        blend_ms: Annotated[
            int,
            Field(
                description="Reserved: blend duration (ms). Accepted but not applied in v1. Range [0,1000].",
                ge=0,
                le=1000,
            ),
        ] = 60,
        space: Annotated[
            str,
            Field(description="Coordinate space name. v1 only supports unity.", examples=["unity"]),
        ] = "unity",
        ctx: Context | None = None,
    ) -> dict:
        """Set a tracker's pose (single update)."""
        # v1: space only supports "unity" (per PLAN)
        if space != "unity":
            trace_id = _trace_id(ctx)
            return _err(
                trace_id=trace_id,
                error_obj={
                    "code": "INVALID_ARGUMENT",
                    "message": "space only supports 'unity'.",
                    "details": {"space": space},
                },
            )
        if blend_ms < 0 or blend_ms > 1000:
            trace_id = _trace_id(ctx)
            return _err(
                trace_id=trace_id,
                error_obj={
                    "code": "INVALID_ARGUMENT",
                    "message": "blend_ms must be in [0,1000].",
                    "details": {"blend_ms": blend_ms},
                },
            )
        # blend_ms currently accepted but not applied (reserved)
        return await _wrap_async(adapter.tracking_set_tracker_pose)(
            tracker=tracker,
            position_m=position_m,
            rotation_euler_deg=rotation_euler_deg,
            ctx=ctx,
        )

    @_tool(
        name="vrc_tracking_set_head_reference",
        description=(
            "Set head reference (optional position/rotation) and alignment mode (single_align/stream_align)."
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_tracking_set_head_reference(
        position_m: Annotated[
            dict | None,
            Field(description="Optional: head reference position in meters. Object: {x,y,z}."),
        ] = None,
        rotation_euler_deg: Annotated[
            dict | None,
            Field(description="Optional: head reference rotation in degrees. Object: {x,y,z}."),
        ] = None,
        mode: Annotated[
            str,
            Field(description="Alignment mode: single_align or stream_align.", examples=["stream_align"]),
        ] = "stream_align",
        ctx: Context | None = None,
    ) -> dict:
        """Set head reference (for tracking alignment/calibration)."""
        return await _wrap_async(adapter.tracking_set_head_reference)(
            position_m=position_m,
            rotation_euler_deg=rotation_euler_deg,
            mode=mode,
            ctx=ctx,
        )

    @_tool(
        name="vrc_tracking_stream_start",
        description="Start the background tracking stream (continuously sends tracker data at fps).",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_tracking_stream_start(
        fps: Annotated[int, Field(description="Stream FPS; allowed values: 20/30/45/60.", examples=[60])] = 60,
        enabled_trackers: Annotated[
            list[str] | None,
            Field(description="Enabled tracker list (up to 8). If omitted, defaults are used."),
        ] = None,
        neutral_on_stop: Annotated[
            bool,
            Field(description="Whether to send a neutral pose on stop (best-effort)."),
        ] = True,
        ctx: Context | None = None,
    ) -> dict:
        """Start the tracking stream (idempotent)."""
        if fps not in (20, 30, 45, 60):
            trace_id = _trace_id(ctx)
            return _err(
                trace_id=trace_id,
                error_obj={
                    "code": "INVALID_ARGUMENT",
                    "message": "fps must be one of 20/30/45/60.",
                    "details": {"fps": fps},
                },
            )
        if enabled_trackers is None:
            enabled_trackers = ["hip", "left_foot", "right_foot"]
        return await _wrap_async(adapter.tracking_stream_start)(
            fps=fps,
            enabled_trackers=enabled_trackers,
            neutral_on_stop=neutral_on_stop,
            ctx=ctx,
        )

    @_tool(
        name="vrc_tracking_stream_stop",
        description="Stop the background tracking stream (idempotent). stream_id is currently reserved.",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_tracking_stream_stop(
        stream_id: Annotated[
            str | None,
            Field(description="Reserved: stream_id to stop. Current implementation ignores this value."),
        ] = None,
        ctx: Context | None = None,
    ) -> dict:
        """Stop the tracking stream."""
        return await _wrap_async(adapter.tracking_stream_stop)(stream_id=stream_id, ctx=ctx)

    # -----------------
    # Eye
    # -----------------

    @_tool(
        name="vrc_eye_stream_start",
        description="Start the background eye tracking stream (continuously sends gaze/blink, etc. at fps).",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_eye_stream_start(
        fps: Annotated[int, Field(description="Stream FPS; allowed values: 20/30/45/60.", examples=[60])] = 60,
        gaze_mode: Annotated[
            str,
            Field(
                description=(
                    "Gaze data format (gaze_mode). Examples: CenterPitchYaw/CenterVec/LeftRightPitchYaw. "
                    "Only one gaze_mode can be used at a time."
                ),
                examples=["CenterPitchYaw"],
            ),
        ] = "CenterPitchYaw",
        neutral_on_stop: Annotated[bool, Field(description="Whether to send neutral gaze/blink on stop (best-effort).")] = True,
        ctx: Context | None = None,
    ) -> dict:
        """Start the eye stream (idempotent)."""
        if fps not in (20, 30, 45, 60):
            trace_id = _trace_id(ctx)
            return _err(
                trace_id=trace_id,
                error_obj={
                    "code": "INVALID_ARGUMENT",
                    "message": "fps must be one of 20/30/45/60.",
                    "details": {"fps": fps},
                },
            )
        return await _wrap_async(adapter.eye_stream_start)(fps=fps, gaze_mode=gaze_mode, neutral_on_stop=neutral_on_stop, ctx=ctx)

    @_tool(
        name="vrc_eye_set_blink",
        description="Set blink/eye-closed amount (EyesClosedAmount). Typical range is 0~1 (not enforced).",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_eye_set_blink(
        amount: Annotated[float, Field(description="Eye-closed amount, typically 0~1. Sent as float.", examples=[0.0, 1.0])],
        ctx: Context | None = None,
    ) -> dict:
        """Set blink and send once immediately."""
        return await _wrap_async(adapter.eye_set_blink)(amount=amount, ctx=ctx)

    @_tool(
        name="vrc_eye_set_gaze",
        description="Set gaze direction/vector (data fields depend on gaze_mode) and send once immediately.",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_eye_set_gaze(
        gaze_mode: Annotated[
            str,
            Field(
                description=(
                    "Gaze data format. Examples: "
                    "CenterPitchYaw -> data={pitch,yaw}; CenterPitchYawDist -> {pitch,yaw,distance or distance_m}; "
                    "CenterVec -> {x,y,z}; LeftRightPitchYaw -> {left_pitch,left_yaw,right_pitch,right_yaw}; "
                    "LeftRightVec -> {left_x,left_y,left_z,right_x,right_y,right_z}."
                ),
                examples=["CenterPitchYaw"],
            ),
        ],
        data: Annotated[dict, Field(description="Data object for the selected gaze_mode (fields must be numbers).")],
        ctx: Context | None = None,
    ) -> dict:
        """Set gaze and send once immediately."""
        return await _wrap_async(adapter.eye_set_gaze)(gaze_mode=gaze_mode, data=data, ctx=ctx)

    @_tool(
        name="vrc_eye_stream_stop",
        description="Stop the background eye tracking stream (idempotent). stream_id is currently reserved.",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_eye_stream_stop(
        stream_id: Annotated[
            str | None,
            Field(description="Reserved: stream_id to stop. Current implementation ignores this value."),
        ] = None,
        ctx: Context | None = None,
    ) -> dict:
        """Stop the eye stream."""
        return await _wrap_async(adapter.eye_stream_stop)(stream_id=stream_id, ctx=ctx)

    # -----------------
    # Chatbox
    # -----------------

    @_tool(
        name="vrc_chatbox_send",
        description="Send Chatbox text (auto-trim to 144 chars / 9 lines, with rate limiting).",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_chatbox_send(
        text: Annotated[
            str,
            Field(
                description=(
                    "Text to send (non-empty). Automatically trimmed to 144 chars / 9 lines to match VRChat limits."
                ),
                min_length=1,
            ),
        ],
        send_immediately: Annotated[
            bool,
            Field(
                description=(
                    "Send immediately: true bypasses the keyboard and sends directly; false opens the keyboard and fills the text. "
                    "When false, notify is ignored (VRChat does not accept the 'n' parameter)."
                )
            ),
        ] = True,
        notify: Annotated[bool, Field(description="Play notification sound (only effective when send_immediately=true).")] = True,
        set_typing: Annotated[bool, Field(description="Set typing=true before sending (cleared after sending, best-effort).")] = False,
        ctx: Context | None = None,
    ) -> dict:
        """Send Chatbox text.

        Semantics:
        - Auto-trim (144 chars / 9 lines)
        - Rate limiting (chat_per_minute)
        - If send_immediately=false, send only (text, False)
        - If send_immediately=true, send (text, True, notify)
        """
        return await _wrap_async(adapter.chatbox_send)(
            text=text,
            send_immediately=send_immediately,
            notify=notify,
            set_typing=set_typing,
            ctx=ctx,
        )

    @_tool(
        name="vrc_chatbox_set_typing",
        description="Set Chatbox typing state (shows \"typing\").",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_chatbox_set_typing(
        is_typing: Annotated[bool, Field(description="Whether the user is typing (true/false).")],
        ctx: Context | None = None,
    ) -> dict:
        """Set /chatbox/typing."""
        return await _wrap_async(adapter.chatbox_set_typing)(is_typing=is_typing, ctx=ctx)

    # -----------------
    # Global safety
    # -----------------

    @_tool(
        name="vrc_stop_all",
        description=(
            "Global emergency stop: stop input/tracking/eye streams and clear chatbox typing (best-effort)."
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_stop_all(ctx: Context | None = None) -> dict:
        """Global stop (idempotent)."""
        return await _wrap_async(adapter.stop_all)(ctx=ctx)

    # -----------------
    # Macro
    # -----------------

    @_tool(
        name="vrc_macro_move_for",
        description=(
            "Macro: move/turn for a duration using given axis values (internally uses vrc_input_set_axes safety auto-zero semantics)."
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_macro_move_for(
        duration_ms: Annotated[int, Field(description="Duration (ms). Range [50,30000].", ge=0)],
        forward: Annotated[float, Field(description="Forward axis (Vertical). Range [-1,1].", examples=[-1.0, 0.0, 1.0])] = 0,
        strafe: Annotated[float, Field(description="Strafe axis (Horizontal). Range [-1,1].", examples=[-1.0, 0.0, 1.0])] = 0,
        turn: Annotated[float, Field(description="Turn axis (LookHorizontal). Range [-1,1].", examples=[-1.0, 0.0, 1.0])] = 0,
        ctx: Context | None = None,
    ) -> dict:
        """Macro: move/turn for a duration (auto-zero enforced)."""
        async def _impl(*, trace_id: str):
            if duration_ms < 50 or duration_ms > 30000:
                raise DomainError(code="INVALID_ARGUMENT", message="duration_ms must be in [50,30000].", details={"duration_ms": duration_ms})
            for name, v in ("forward", forward), ("strafe", strafe), ("turn", turn):
                if v < -1 or v > 1:
                    raise DomainError(code="INVALID_ARGUMENT", message=f"{name} must be in [-1,1].", details={name: v})

            # Only include axes the caller actually wants to drive.
            # This avoids unintentionally clobbering other concurrent inputs
            # (e.g., turning via another tool while this macro only moves forward).
            axes: dict[str, float] = {}
            if forward != 0:
                axes["Vertical"] = float(forward)
            if strafe != 0:
                axes["Horizontal"] = float(strafe)
            if turn != 0:
                axes["LookHorizontal"] = float(turn)

            if not axes:
                return {"no_op": True, "reason": "all_axes_zero"}

            return await adapter.input_set_axes(
                axes=axes,
                duration_ms=duration_ms,
                auto_zero=True,
                ease_ms=80,
                trace_id=trace_id,
            )

        return await _wrap_async(_impl)(ctx=ctx)

    @_tool(
        name="vrc_macro_turn_degrees",
        description=(
            "Macro: convert degrees into a short LookHorizontal pulse using a conservative heuristic mapping."
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_macro_turn_degrees(
        degrees: Annotated[float, Field(description="Desired turn angle in degrees. Positive turns right, negative turns left.", examples=[-90.0, 90.0])],
        ctx: Context | None = None,
    ) -> dict:
        """Macro: approximately turn by a specified angle."""
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

    @_tool(
        name="vrc_macro_look_at",
        description=(
            "Macro: look towards a target yaw/pitch (prefer eye; fall back to input on failure)."
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_macro_look_at(
        yaw_deg: Annotated[float, Field(description="Yaw angle in degrees. Recommended range [-90,90].", examples=[-30.0, 30.0])],
        pitch_deg: Annotated[float, Field(description="Pitch angle in degrees. Recommended range [-45,45].", examples=[-10.0, 10.0])],
        duration_ms: Annotated[int, Field(description="Hold duration (ms). Range [50,10000].", ge=0)] = 800,
        prefer_eye: Annotated[bool, Field(description="Prefer eye gaze if available (falls back to input automatically).")] = True,
        ctx: Context | None = None,
    ) -> dict:
        """Macro: look towards a target direction for a duration."""
        async def _impl(*, trace_id: str):
            if yaw_deg < -90 or yaw_deg > 90:
                raise DomainError(code="INVALID_ARGUMENT", message="yaw_deg must be in [-90,90].", details={"yaw_deg": yaw_deg})
            if pitch_deg < -45 or pitch_deg > 45:
                raise DomainError(code="INVALID_ARGUMENT", message="pitch_deg must be in [-45,45].", details={"pitch_deg": pitch_deg})
            if duration_ms < 50 or duration_ms > 10000:
                raise DomainError(code="INVALID_ARGUMENT", message="duration_ms must be in [50,10000].", details={"duration_ms": duration_ms})
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

    @_tool(
        name="vrc_macro_emote",
        description="Macro: trigger an emote (v1 requires an emote mapping table; currently a placeholder).",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    def vrc_macro_emote(
        name: Annotated[str, Field(description="Emote name (must be mapped in config).", min_length=1)],
        ctx: Context | None = None,
    ) -> dict:
        """Placeholder: emote requires a profile/emote mapping table."""
        # Placeholder: v1 requires a profile/emote mapping table.
        trace_id = _trace_id(ctx)
        return _err(
            trace_id=trace_id,
            error_obj={
                "code": "CAPABILITY_UNAVAILABLE",
                "message": "macro_emote is not configured (requires an emote-name -> avatar parameter/action mapping table).",
                "details": {"name": name},
            },
        )

    @_tool(
        name="vrc_macro_idle",
        description="Macro: enter idle (stop movement input and clear typing).",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_macro_idle(ctx: Context | None = None) -> dict:
        """Macro: stop input and set chatbox typing to false."""
        async def _impl(*, trace_id: str):
            await adapter.input_stop(trace_id=trace_id)
            await adapter.chatbox_set_typing(is_typing=False, trace_id=trace_id)
            return {"idle": True}

        return await _wrap_async(_impl)(ctx=ctx)

    @_tool(
        name="vrc_macro_stop",
        description="Macro: stop everything (equivalent to vrc_stop_all).",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_macro_stop(ctx: Context | None = None) -> dict:
        """Macro stop (calls stop_all directly)."""
        # Per PLAN: macro_stop calls stop_all directly
        return await _wrap_async(adapter.stop_all)(ctx=ctx)

    return mcp

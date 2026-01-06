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
        description="获取服务状态（是否检测到 OSC、目标地址、上次发送时间、tracking/eye 流状态）。",
        annotations={"readOnlyHint": True},
        output_schema=_STATUS_OUTPUT_SCHEMA,
    )
    def vrc_meta_get_status() -> dict:
        """查询 vrchat-osc-mcp 当前状态。

        用途：让调用方判断 VRChat/OSC 是否“看起来可用”，以及后台 tracking/eye stream 是否在运行。
        """
        return _wrap_sync(lambda *, trace_id: adapter.meta_get_status())()

    @_tool(
        name="vrc_meta_get_capabilities",
        description="获取能力清单（支持的 input 轴/按钮、tracking/eye/chatbox 可用性、avatar schema 置信度等）。",
        annotations={"readOnlyHint": True},
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    def vrc_meta_get_capabilities(
        refresh: Annotated[
            bool,
            Field(
                description="是否刷新能力/缓存（v1 预留字段；当前实现为 best-effort）。",
            ),
        ] = False,
    ) -> dict:
        """获取当前服务的能力与提示信息。

        建议在做动作前先读 capabilities，避免 LLM “瞎按键”。
        """
        return _wrap_sync(lambda refresh, *, trace_id: adapter.meta_get_capabilities(refresh=refresh))(refresh)

    # -----------------
    # Input
    # -----------------

    @_tool(
        name="vrc_input_list_endpoints",
        description="列出本服务器 v1 对外支持的 /input 轴与按钮名称（建议先调用再传参）。",
        annotations={"readOnlyHint": True},
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    def vrc_input_list_endpoints() -> dict:
        """列出支持的 input endpoints（axes/buttons）。"""
        return _wrap_sync(lambda *, trace_id: adapter.input_list_endpoints())()

    @_tool(
        name="vrc_input_set_axes",
        description=(
            "设置 /input 轴（例如移动/转向）。值会自动夹紧到 [-1,1]，并且无论 auto_zero 与否都会强制有界归零，"
            "避免持续输入。"
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_input_set_axes(
        axes: Annotated[
            dict,
            Field(
                description=(
                    "要设置的轴值对象。key 为轴名（v1: Vertical/Horizontal/LookHorizontal），value 为 number。"
                    "超出 [-1,1] 会被自动夹紧；未知轴名会报错。"
                ),
                examples=[{"Vertical": 1.0}],
            ),
        ],
        duration_ms: Annotated[
            int,
            Field(
                description=(
                    "持续时间（毫秒）。<=0 时将使用 ease_ms 作为默认持续时间；最终会被限制到 safety.max_axis_duration_ms。"
                    "注意：为安全起见工具会强制归零（enforced_auto_zero）。"
                ),
                ge=0,
            ),
        ] = 0,
        auto_zero: Annotated[
            bool,
            Field(
                description=(
                    "是否请求自动归零（requested_auto_zero）。即便传 false，服务器仍会强制有界归零以避免“永动机式前进”。"
                )
            ),
        ] = True,
        ease_ms: Annotated[
            int,
            Field(
                description="缓动/插值时长（毫秒）。当 duration_ms<=0 时作为默认持续时间；范围 [0,500]。",
                ge=0,
                le=500,
            ),
        ] = 80,
        ctx: Context | None = None,
    ) -> dict:
        """设置一个或多个 /input 轴。

        关键语义（安全阀）：
        - 轴值会被夹紧到 [-1,1]
        - 最终 duration 会被限制到 settings.safety.max_axis_duration_ms
        - 无论 auto_zero 与否都会强制归零（以避免持续输入）
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
        description="点按按钮（自动 press + release）。用于 Jump/Use 等瞬时动作；避免长按副作用。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_input_tap_buttons(
        buttons: Annotated[
            list[str],
            Field(
                description="要点按的按钮名数组（建议先 vrc_input_list_endpoints）。最多 10 个。",
                min_length=1,
                max_length=10,
            ),
        ],
        press_ms: Annotated[
            int,
            Field(
                description="按住时长（毫秒）。将被钳制到 [20,1000] 以提高 VRChat 识别概率。",
                ge=0,
                le=1000,
            ),
        ] = 80,
        ctx: Context | None = None,
    ) -> dict:
        """点按一个或多个 /input 按钮。

        语义：每个按钮都会发送 1，然后在 finally 中保证发送 0（即使中途异常）。
        """
        return await _wrap_async(adapter.input_tap_buttons)(buttons=buttons, press_ms=press_ms, ctx=ctx)

    @_tool(
        name="vrc_input_hold_buttons",
        description="按住按钮（发送 value=1 并保持）。需要后续调用 vrc_input_release_buttons 或 vrc_input_stop 释放。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_input_hold_buttons(
        buttons: Annotated[
            list[str],
            Field(
                description="要按住的按钮名数组（最多 10 个）。",
                min_length=1,
                max_length=10,
            ),
        ],
        ctx: Context | None = None,
    ) -> dict:
        """按住（hold）按钮。"""
        return await _wrap_async(adapter.input_hold_buttons)(buttons=buttons, ctx=ctx)

    @_tool(
        name="vrc_input_release_buttons",
        description="释放按钮（发送 value=0）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_input_release_buttons(
        buttons: Annotated[
            list[str],
            Field(
                description="要释放的按钮名数组（最多 10 个）。",
                min_length=1,
                max_length=10,
            ),
        ],
        ctx: Context | None = None,
    ) -> dict:
        """释放（release）按钮。"""
        return await _wrap_async(adapter.input_release_buttons)(buttons=buttons, ctx=ctx)

    @_tool(
        name="vrc_input_stop",
        description="Input 子系统急停：所有轴归零、释放所有按钮（幂等）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_input_stop(ctx: Context | None = None) -> dict:
        """停止所有 /input 影响（归零 + 释放）。"""
        return await _wrap_async(adapter.input_stop)(ctx=ctx)

    # -----------------
    # Avatar
    # -----------------

    @_tool(
        name="vrc_avatar_list_parameters",
        description="列出当前 Avatar 参数（基于本地 OSC config/schema，若未加载则可能为空）。",
        annotations={"readOnlyHint": True},
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    def vrc_avatar_list_parameters() -> dict:
        """列出当前 avatar 的参数清单（若 schema 未加载则返回空）。"""
        return _wrap_sync(lambda *, trace_id: adapter.avatar_list_parameters())()

    @_tool(
        name="vrc_avatar_set_parameter",
        description=(
            "设置单个 Avatar 参数。支持 duration_ms + reset_value 做脉冲；参数策略由 safety.parameter_policy 控制。"
        ),
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_avatar_set_parameter(
        name: Annotated[
            str,
            Field(description="参数名（推荐先 vrc_avatar_list_parameters；也可直接传 VRChat 参数名）。", min_length=1),
        ],
        value: Any,
        duration_ms: Annotated[
            int,
            Field(description="可选：持续时间（毫秒）。>0 时在等待后自动写回 reset_value/默认值。", ge=0),
        ] = 0,
        reset_value: Any = None,
        ctx: Context | None = None,
    ) -> dict:
        """设置单个 Avatar 参数。

        说明：
        - value / reset_value 最佳实践是 bool/int/float（VRChat 参数类型）。
        - strict 策略下会校验 schema 是否存在且参数可写。
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
        description="批量设置多个 Avatar 参数（减少 tool call）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_avatar_set_parameters(
        parameters: Annotated[
            list[dict],
            Field(description="参数对象数组：[{name, value}, ...]。至少 1 个，最多建议 64 个。", min_length=1),
        ],
        ctx: Context | None = None,
    ) -> dict:
        """批量设置参数（内部逐个发送，返回每项结果）。"""
        return await _wrap_async(adapter.avatar_set_parameters)(parameters=parameters, ctx=ctx)

    # -----------------
    # Tracking
    # -----------------

    @_tool(
        name="vrc_tracking_set_tracker_pose",
        description="设置某个 tracker 的目标位姿（position_m / rotation_euler_deg）。通常配合 tracking stream 使用。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_tracking_set_tracker_pose(
        tracker: Annotated[
            str,
            Field(
                description=(
                    "Tracker 名称（例如 hip/left_foot/right_foot 等）。"
                    "建议使用 vrc_meta_get_capabilities 了解推荐 tracker。"
                ),
                min_length=1,
            ),
        ],
        position_m: Annotated[
            dict,
            Field(description="位置（米）。对象：{x,y,z}。坐标系按 Unity 假设。"),
        ],
        rotation_euler_deg: Annotated[
            dict,
            Field(description="旋转欧拉角（度）。对象：{x,y,z}。"),
        ],
        blend_ms: Annotated[
            int,
            Field(description="预留：混合时长（毫秒）。v1 接受但不应用。范围 [0,1000]。", ge=0, le=1000),
        ] = 60,
        space: Annotated[
            str,
            Field(description="坐标系名称。v1 仅支持 unity。", examples=["unity"]),
        ] = "unity",
        ctx: Context | None = None,
    ) -> dict:
        """设置 tracker 位姿（单次更新）。"""
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

    @_tool(
        name="vrc_tracking_set_head_reference",
        description="设置 head reference（可选 position/rotation）以及对齐模式（single_align/stream_align）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_tracking_set_head_reference(
        position_m: Annotated[
            dict | None,
            Field(description="可选：头部参考位置（米）。对象：{x,y,z}。"),
        ] = None,
        rotation_euler_deg: Annotated[
            dict | None,
            Field(description="可选：头部参考旋转（度）。对象：{x,y,z}。"),
        ] = None,
        mode: Annotated[
            str,
            Field(description="对齐模式：single_align 或 stream_align。", examples=["stream_align"]),
        ] = "stream_align",
        ctx: Context | None = None,
    ) -> dict:
        """设置 head reference（用于追踪对齐/校准）。"""
        return await _wrap_async(adapter.tracking_set_head_reference)(
            position_m=position_m,
            rotation_euler_deg=rotation_euler_deg,
            mode=mode,
            ctx=ctx,
        )

    @_tool(
        name="vrc_tracking_stream_start",
        description="启动 tracking 后台流（按 fps 持续发送 tracker 数据）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_tracking_stream_start(
        fps: Annotated[int, Field(description="流发送帧率，只允许 20/30/45/60。", examples=[60])] = 60,
        enabled_trackers: Annotated[
            list[str] | None,
            Field(description="启用的 tracker 列表（最多 8 个）。省略则使用默认值。"),
        ] = None,
        neutral_on_stop: Annotated[
            bool,
            Field(description="停止时是否发送中立姿态（best-effort）。"),
        ] = True,
        ctx: Context | None = None,
    ) -> dict:
        """启动 tracking stream（幂等）。"""
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

    @_tool(
        name="vrc_tracking_stream_stop",
        description="停止 tracking 后台流（幂等）。stream_id 当前为预留字段。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_tracking_stream_stop(
        stream_id: Annotated[str | None, Field(description="预留：要停止的 stream_id。当前实现忽略该值。")] = None,
        ctx: Context | None = None,
    ) -> dict:
        """停止 tracking stream。"""
        return await _wrap_async(adapter.tracking_stream_stop)(stream_id=stream_id, ctx=ctx)

    # -----------------
    # Eye
    # -----------------

    @_tool(
        name="vrc_eye_stream_start",
        description="启动 eye tracking 后台流（按 fps 持续发送 gaze/眨眼等）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_eye_stream_start(
        fps: Annotated[int, Field(description="流发送帧率，只允许 20/30/45/60。", examples=[60])] = 60,
        gaze_mode: Annotated[
            str,
            Field(
                description=(
                    "注视数据格式（gaze_mode）。例如 CenterPitchYaw/CenterVec/LeftRightPitchYaw 等。"
                    "同一时间只能使用一种 gaze_mode。"
                ),
                examples=["CenterPitchYaw"],
            ),
        ] = "CenterPitchYaw",
        neutral_on_stop: Annotated[bool, Field(description="停止时是否发送中立 gaze/眨眼（best-effort）。")] = True,
        ctx: Context | None = None,
    ) -> dict:
        """启动 eye stream（幂等）。"""
        if fps not in (20, 30, 45, 60):
            trace_id = _trace_id(ctx)
            return _err(
                trace_id=trace_id,
                error_obj={"code": "INVALID_ARGUMENT", "message": "fps 必须是 20/30/45/60。", "details": {"fps": fps}},
            )
        return await _wrap_async(adapter.eye_stream_start)(fps=fps, gaze_mode=gaze_mode, neutral_on_stop=neutral_on_stop, ctx=ctx)

    @_tool(
        name="vrc_eye_set_blink",
        description="设置眨眼/闭眼程度（EyesClosedAmount）。通常范围 0~1（不强制）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_eye_set_blink(
        amount: Annotated[float, Field(description="闭眼程度，通常 0~1。会被发送为 float。", examples=[0.0, 1.0])],
        ctx: Context | None = None,
    ) -> dict:
        """设置眨眼（并立即发送一次）。"""
        return await _wrap_async(adapter.eye_set_blink)(amount=amount, ctx=ctx)

    @_tool(
        name="vrc_eye_set_gaze",
        description="设置注视方向/向量（按 gaze_mode 决定 data 字段）。并立即发送一次。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_eye_set_gaze(
        gaze_mode: Annotated[
            str,
            Field(
                description=(
                    "注视数据格式。示例："
                    "CenterPitchYaw -> data={pitch,yaw}; CenterPitchYawDist -> {pitch,yaw,distance或distance_m}; "
                    "CenterVec -> {x,y,z}; LeftRightPitchYaw -> {left_pitch,left_yaw,right_pitch,right_yaw}; "
                    "LeftRightVec -> {left_x,left_y,left_z,right_x,right_y,right_z}。"
                ),
                examples=["CenterPitchYaw"],
            ),
        ],
        data: Annotated[dict, Field(description="gaze_mode 对应的数据对象（字段必须是 number）。")],
        ctx: Context | None = None,
    ) -> dict:
        """设置 gaze（并立即发送一次）。"""
        return await _wrap_async(adapter.eye_set_gaze)(gaze_mode=gaze_mode, data=data, ctx=ctx)

    @_tool(
        name="vrc_eye_stream_stop",
        description="停止 eye tracking 后台流（幂等）。stream_id 当前为预留字段。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_eye_stream_stop(
        stream_id: Annotated[str | None, Field(description="预留：要停止的 stream_id。当前实现忽略该值。")] = None,
        ctx: Context | None = None,
    ) -> dict:
        """停止 eye stream。"""
        return await _wrap_async(adapter.eye_stream_stop)(stream_id=stream_id, ctx=ctx)

    # -----------------
    # Chatbox
    # -----------------

    @_tool(
        name="vrc_chatbox_send",
        description="发送 Chatbox 文本（自动裁剪到 144 字符 / 9 行，并带频率限制）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_chatbox_send(
        text: Annotated[
            str,
            Field(
                description=(
                    "要发送的文本（非空）。将自动裁剪到 144 字符 / 9 行以符合 VRChat 限制。"
                ),
                min_length=1,
            ),
        ],
        send_immediately: Annotated[
            bool,
            Field(
                description=(
                    "是否立即发送：true 表示 bypass 键盘并直接发送；false 表示打开键盘并填充文本。"
                    "当为 false 时，notify 参数会被忽略（VRChat 端不接收 n）。"
                )
            ),
        ] = True,
        notify: Annotated[bool, Field(description="是否播放通知音效（仅在 send_immediately=true 时有效）。")] = True,
        set_typing: Annotated[bool, Field(description="是否在发送前设置 typing=true（发送后会自动清回，best-effort）。")] = False,
        ctx: Context | None = None,
    ) -> dict:
        """发送 chatbox 文本。

        语义：
        - 自动裁剪（144 chars / 9 lines）
        - 限流（chat_per_minute）
        - send_immediately=false 时仅发送 (text, False)
        - send_immediately=true 时发送 (text, True, notify)
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
        description="设置 Chatbox typing 状态（显示“正在输入”）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_chatbox_set_typing(
        is_typing: Annotated[bool, Field(description="是否处于输入中（true/false）。")],
        ctx: Context | None = None,
    ) -> dict:
        """设置 /chatbox/typing。"""
        return await _wrap_async(adapter.chatbox_set_typing)(is_typing=is_typing, ctx=ctx)

    # -----------------
    # Global safety
    # -----------------

    @_tool(
        name="vrc_stop_all",
        description="全局急停：停止 input/tracking/eye 流，并清除 chatbox typing（best-effort）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_stop_all(ctx: Context | None = None) -> dict:
        """全局 stop（幂等）。"""
        return await _wrap_async(adapter.stop_all)(ctx=ctx)

    # -----------------
    # Macro
    # -----------------

    @_tool(
        name="vrc_macro_move_for",
        description="宏：按给定轴值移动/转向一段时间（内部使用 vrc_input_set_axes 的安全归零语义）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_macro_move_for(
        duration_ms: Annotated[int, Field(description="持续时间（毫秒）。范围 [50,30000]。", ge=0)],
        forward: Annotated[float, Field(description="前进轴（Vertical）。范围 [-1,1]。", examples=[-1.0, 0.0, 1.0])] = 0,
        strafe: Annotated[float, Field(description="横移轴（Horizontal）。范围 [-1,1]。", examples=[-1.0, 0.0, 1.0])] = 0,
        turn: Annotated[float, Field(description="转向轴（LookHorizontal）。范围 [-1,1]。", examples=[-1.0, 0.0, 1.0])] = 0,
        ctx: Context | None = None,
    ) -> dict:
        """宏：移动/转向一段时间（自动归零）。"""
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

    @_tool(
        name="vrc_macro_turn_degrees",
        description="宏：按近似映射将 degrees 转为 LookHorizontal 的短脉冲转向（保守估计）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_macro_turn_degrees(
        degrees: Annotated[float, Field(description="期望转向角度（度）。正值向右，负值向左。", examples=[-90.0, 90.0])],
        ctx: Context | None = None,
    ) -> dict:
        """宏：近似转向指定角度。"""
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
        description="宏：让视线朝向指定 yaw/pitch（优先使用 eye；失败则回退到 input）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_macro_look_at(
        yaw_deg: Annotated[float, Field(description="偏航角 yaw（度）。范围建议 [-90,90]。", examples=[-30.0, 30.0])],
        pitch_deg: Annotated[float, Field(description="俯仰角 pitch（度）。范围建议 [-45,45]。", examples=[-10.0, 10.0])],
        duration_ms: Annotated[int, Field(description="保持时长（毫秒）。范围 [50,10000]。", ge=0)] = 800,
        prefer_eye: Annotated[bool, Field(description="是否优先使用 eye gaze（若不可用将自动回退到 input）。")] = True,
        ctx: Context | None = None,
    ) -> dict:
        """宏：注视目标方向一段时间。"""
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

    @_tool(
        name="vrc_macro_emote",
        description="宏：触发 emote（v1 需要配置 emote 映射表；当前为占位符）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    def vrc_macro_emote(name: Annotated[str, Field(description="Emote 名称（需要在配置中映射）。", min_length=1)], ctx: Context | None = None) -> dict:
        """占位符：emote 需要 profile/emote 映射表。"""
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

    @_tool(
        name="vrc_macro_idle",
        description="宏：进入 idle（停止移动输入，并清除 typing）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_macro_idle(ctx: Context | None = None) -> dict:
        """宏：停止 input，并将 chatbox typing 置为 false。"""
        async def _impl(*, trace_id: str):
            await adapter.input_stop(trace_id=trace_id)
            await adapter.chatbox_set_typing(is_typing=False, trace_id=trace_id)
            return {"idle": True}

        return await _wrap_async(_impl)(ctx=ctx)

    @_tool(
        name="vrc_macro_stop",
        description="宏：停止所有动作（等同 vrc_stop_all）。",
        output_schema=_ENVELOPE_OUTPUT_SCHEMA,
    )
    async def vrc_macro_stop(ctx: Context | None = None) -> dict:
        """宏 stop（直接调用 stop_all）。"""
        # Per PLAN: macro_stop 直接调用 stop_all
        return await _wrap_async(adapter.stop_all)(ctx=ctx)

    return mcp

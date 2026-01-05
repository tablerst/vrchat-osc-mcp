from __future__ import annotations

import re
import uuid

from fastmcp import Context, FastMCP

from .domain.errors import DomainError

# OpenAI function/tool name constraints (used by some MCP clients):
# - Allowed chars: A-Z a-z 0-9 _ -
# - Length capped (historically 64)
_OPENAI_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Public MCP tool names exposed by this server.
# NOTE: We intentionally avoid dots (.) to stay compatible with OpenAI tool naming rules.
VRC_TOOL_NAMES = {
    "status": "vrc_status",
    "list_inputs": "vrc_list_inputs",
    "list_parameters": "vrc_list_parameters",
    "set_parameter": "vrc_set_parameter",
    "input_tap": "vrc_input_tap",
    "input_axis": "vrc_input_axis",
    "chat_send": "vrc_chat_send",
    "chat_typing": "vrc_chat_typing",
}


def create_server(*, adapter) -> FastMCP:
    mcp = FastMCP(
        name="vrchat-osc-mcp",
        # Avoid leaking internals by default; our DomainError payload is explicit anyway.
        mask_error_details=True,
    )

    def _tool(*, name: str, **kwargs):
        # Fail fast if someone adds an incompatible name.
        if not _OPENAI_TOOL_NAME_RE.fullmatch(name):
            raise ValueError(
                f"Invalid MCP tool name: {name!r}. "
                "Tool names must match ^[A-Za-z0-9_-]{1,64}$."
            )
        return mcp.tool(name=name, **kwargs)

    @_tool(name=VRC_TOOL_NAMES["status"], annotations={"readOnlyHint": True})
    def vrc_status() -> dict:
        """返回服务器运行状态、OSC目标与安全阀配置摘要。"""

        return adapter.status()

    @_tool(name=VRC_TOOL_NAMES["list_inputs"], annotations={"readOnlyHint": True})
    def vrc_list_inputs() -> dict:
        """列出 VRChat 官方 OSC /input 支持的轴与按钮，以及它们的作用说明。

        本服务对 vrc_input_tap / vrc_input_axis 做了“只允许官方清单”的校验：
        - 传错名称会返回 E_INPUT_UNKNOWN，并附带 supported 列表
        - 名称大小写不敏感（例如 "jump" 会被规范化为 "Jump"）
        """

        return adapter.list_inputs()

    @_tool(name=VRC_TOOL_NAMES["list_parameters"], annotations={"readOnlyHint": True})
    def vrc_list_parameters() -> dict:
        """列出当前 Avatar schema 中的参数能力表（若已加载）。"""

        return adapter.list_parameters()

    @_tool(name=VRC_TOOL_NAMES["set_parameter"])
    async def vrc_set_parameter(name: str, value: bool | int | float, ctx: Context, ttl_ms: int = 0) -> dict:
        """写入 Avatar 参数：/avatar/parameters/<name>。

        安全阀：MVP-0 默认 allowlist 拒绝未知参数。
        """

        trace_id = ctx.request_id or str(uuid.uuid4())
        try:
            return await adapter.set_parameter(name=name, value=value, ttl_ms=ttl_ms, trace_id=trace_id)
        except DomainError as e:
            return e.to_dict()

    @_tool(name=VRC_TOOL_NAMES["input_tap"])
    async def vrc_input_tap(button: str, ctx: Context, hold_ms: int = 80) -> dict:
        """按键点按（/input/<Button>）：发送 1，等待 hold_ms，再发送 0（强制复位）。

        仅支持 VRChat 官方按钮名（可调用 vrc_list_inputs 查看清单与作用）。

        备注：部分按钮是“仅 VR”或“依赖世界/设置”的行为（例如 Jump/Run/Voice）。
        """

        trace_id = ctx.request_id or str(uuid.uuid4())
        try:
            return await adapter.input_tap(button=button, hold_ms=hold_ms, trace_id=trace_id)
        except DomainError as e:
            return e.to_dict()

    @_tool(name=VRC_TOOL_NAMES["input_axis"])
    async def vrc_input_axis(axis: str, value: float, duration_ms: int, ctx: Context) -> dict:
        """轴输入（/input/<Axis>）：发送 value（clamp 到 [-1,1]），等待 duration_ms，再发送 0.0（强制复位）。

        仅支持 VRChat 官方轴名（可调用 vrc_list_inputs 查看清单与作用）。

        - value：会被 clamp 到 [-1, 1]
        - duration_ms：必填；并会被安全阀截断到配置的上限
        """

        trace_id = ctx.request_id or str(uuid.uuid4())
        try:
            return await adapter.input_axis(axis=axis, value=value, duration_ms=duration_ms, trace_id=trace_id)
        except DomainError as e:
            return e.to_dict()

    @_tool(name=VRC_TOOL_NAMES["chat_send"])
    async def vrc_chat_send(text: str, ctx: Context, immediate: bool = True, notify: bool = True) -> dict:
        """发送 Chatbox 文本（/chatbox/input）。

        VRChat 官方格式：/chatbox/input s b n

        - text：要发送的文本（会自动裁剪到 144 字 / 9 行）
        - immediate（b）：True=绕过键盘立即发送；False=打开键盘并填入文本
        - notify（n）：False=不触发通知音效；True=正常通知

        额外安全阀：本服务对 chatbox 发送做了限流，过快会返回 E_RATE_LIMITED。
        """

        trace_id = ctx.request_id or str(uuid.uuid4())
        try:
            return await adapter.chat_send(text=text, immediate=immediate, notify=notify, trace_id=trace_id)
        except DomainError as e:
            return e.to_dict()

    @_tool(name=VRC_TOOL_NAMES["chat_typing"])
    async def vrc_chat_typing(on: bool, ctx: Context) -> dict:
        """设置 Chatbox typing 指示（/chatbox/typing）。

        VRChat 官方格式：/chatbox/typing b
        - on=True 打开；on=False 关闭
        """

        trace_id = ctx.request_id or str(uuid.uuid4())
        try:
            return await adapter.chat_typing(on=on, trace_id=trace_id)
        except DomainError as e:
            return e.to_dict()

    return mcp

from __future__ import annotations

import asyncio
import time
from typing import Any

from .chatbox import trim_chatbox_text
from .errors import DomainError, ok
from .safety import SlidingWindowRateLimiter
from ..vrc_config.schema import AvatarSchema, ParameterType


def _clamp_axis(v: float) -> tuple[float, bool]:
    if v < -1.0:
        return -1.0, True
    if v > 1.0:
        return 1.0, True
    return v, False


# VRChat 官方 OSC 输入清单（见 doc.md）。
# 我们在 domain 层做校验：工具会明确“仅支持这些”，传错时给出可解释错误。
_INPUT_AXES: dict[str, str] = {
    "Vertical": "向前移动(1) / 向后移动(-1)",
    "Horizontal": "向右移动(1) / 向左移动(-1)",
    "LookHorizontal": "向左/向右看；桌面模式下可用于平滑转向；开启舒适转向时 VR 值为 1 会快速转向",
    "UseAxisRight": "使用手上的物品（不一定有效）",
    "GrabAxisRight": "抓取物品（不一定有效）",
    "MoveHoldFB": "向前(1) / 向后(-1) 移动所持对象",
    "SpinHoldCwCcw": "顺时针/逆时针旋转所持对象",
    "SpinHoldUD": "向上/向下旋转所持对象",
    "SpinHoldLR": "向左/向右旋转所持对象",
}

_INPUT_BUTTONS: dict[str, str] = {
    "MoveForward": "值为 1 时向前移动",
    "MoveBackward": "值为 1 时向后移动",
    "MoveLeft": "值为 1 时向左移动",
    "MoveRight": "值为 1 时向右移动",
    "LookLeft": "值为 1 时向左转动（桌面平滑；VR 舒适转向时快转）",
    "LookRight": "值为 1 时向右转动（桌面平滑；VR 舒适转向时快转）",
    "Jump": "跳跃（世界支持跳跃时有效）",
    "Run": "奔跑（世界支持奔跑时有效）",
    "ComfortLeft": "向左快转（仅 VR）",
    "ComfortRight": "向右快转（仅 VR）",
    "DropRight": "丢掉右手所持物品（仅 VR）",
    "UseRight": "使用被右手高亮显示的物品（仅 VR）",
    "GrabRight": "抓取被右手高亮显示的物品（仅 VR）",
    "DropLeft": "丢掉左手所持物品（仅 VR）",
    "UseLeft": "使用被左手高亮显示的物品（仅 VR）",
    "GrabLeft": "抓取被左手高亮显示的物品（仅 VR）",
    "PanicButton": "打开安全模式",
    "QuickMenuToggleLeft": "切换快捷菜单：从 0->1 会触发切换",
    "QuickMenuToggleRight": "切换快捷菜单：从 0->1 会触发切换",
    "Voice": "语音开关/静音（行为取决于 VRChat 的“切换语音”设置）",
}


def _normalize_input_name(raw: str) -> str:
    # Be forgiving with leading/trailing spaces.
    name = (raw or "").strip()
    # Allow users to paste a full OSC address like "/input/Jump".
    if name.startswith("/input/"):
        name = name.removeprefix("/input/")
    return name


def _resolve_supported_input(*, raw: str, allowed: dict[str, str], kind: str) -> str:
    name = _normalize_input_name(raw)
    if not name:
        raise DomainError(
            code="E_INPUT_INVALID",
            message=f"{kind} 不能为空。",
        )

    if name in allowed:
        return name

    # Case-insensitive convenience (common when LLMs output lowercased names).
    lower = name.lower()
    by_lower = {k.lower(): k for k in allowed}
    if lower in by_lower:
        return by_lower[lower]

    raise DomainError(
        code="E_INPUT_UNKNOWN",
        message=(
            f"未知/不支持的 VRChat /input {kind}: {raw!r}。"
            "请使用 vrc_list_inputs 查看支持清单，或按 VRChat 官方输入名传参。"
        ),
        details={
            "kind": kind,
            "given": raw,
            "normalized": name,
            "supported": sorted(allowed.keys()),
        },
    )


class VRChatDomainAdapter:
    """VRChat semantics + safety valves.

    Non-negotiables (per PLAN/AGENTS):
    - /input/* must auto-reset (finally + bounded duration)
    - axis values clamp to [-1, 1]
    - chatbox trimmed to 144 chars / 9 lines
    """

    def __init__(
        self,
        *,
        transport,
        settings,
        logger,
        schema: AvatarSchema | None = None,
    ) -> None:
        self._transport = transport
        self._settings = settings
        self._logger = logger
        self._schema = schema
        self._chat_limiter = SlidingWindowRateLimiter(
            max_events=settings.safety.chat_per_minute,
            window_s=60.0,
        )

    def list_parameters(self) -> dict[str, Any]:
        if self._schema is None:
            return ok({"schema_loaded": False, "parameters": []})

        params = []
        for p in self._schema.parameters.values():
            params.append(
                {
                    "name": p.name,
                    "type": p.type,
                    "writable": p.writable,
                    "input_address": p.input_address,
                    "output_address": p.output_address,
                }
            )

        params.sort(key=lambda x: x["name"])
        return ok(
            {
                "schema_loaded": True,
                "avatar_id": self._schema.avatar_id,
                "avatar_name": self._schema.avatar_name,
                "schema_path": str(self._schema.source_path) if self._schema.source_path else None,
                "parameters": params,
            }
        )

    def list_inputs(self) -> dict[str, Any]:
        """List supported VRChat official OSC /input controls.

        This is intentionally schema-free (not avatar-specific).
        """

        axes = [
            {
                "name": name,
                "osc_address": f"/input/{name}",
                "kind": "axis",
                "value_range": [-1.0, 1.0],
                "description": desc,
            }
            for name, desc in sorted(_INPUT_AXES.items(), key=lambda kv: kv[0])
        ]
        buttons = [
            {
                "name": name,
                "osc_address": f"/input/{name}",
                "kind": "button",
                "value": "0/1 (本服务会以 tap 形式发送 1 再自动复位 0)",
                "description": desc,
            }
            for name, desc in sorted(_INPUT_BUTTONS.items(), key=lambda kv: kv[0])
        ]

        return ok(
            {
                "axes": axes,
                "buttons": buttons,
                "notes": {
                    "axis_behavior": "vrc_input_axis 会发送 value（clamp 到 [-1,1]），等待 duration_ms，然后发送 0.0 复位。",
                    "button_behavior": "vrc_input_tap 会发送 1，等待 hold_ms，然后发送 0 复位（即使取消/异常也会 finally 复位）。",
                },
            }
        )

    @staticmethod
    def _validate_value(expected: ParameterType, value: bool | int | float) -> tuple[bool | int | float, bool]:
        """Return (coerced_value, coerced_flag) or raise DomainError."""

        if expected == "Bool":
            if isinstance(value, bool):
                return value, False
            raise DomainError(code="E_PARAMETER_TYPE_MISMATCH", message="参数类型为 Bool，但 value 不是 bool。")

        if expected == "Int":
            # bool is a subclass of int in Python, so exclude it explicitly.
            if isinstance(value, bool) or not isinstance(value, int):
                raise DomainError(code="E_PARAMETER_TYPE_MISMATCH", message="参数类型为 Int，但 value 不是 int。")
            return value, False

        # Float
        if isinstance(value, bool):
            raise DomainError(code="E_PARAMETER_TYPE_MISMATCH", message="参数类型为 Float，但 value 不是 number。")
        if isinstance(value, (int, float)):
            coerced = float(value)
            return coerced, coerced != value

        raise DomainError(code="E_PARAMETER_TYPE_MISMATCH", message="value 必须是 bool/int/float。")

    def status(self) -> dict[str, Any]:
        schema_info: dict[str, Any]
        if self._schema is None:
            schema_info = {"source": "unknown"}
        else:
            schema_info = {
                "source": "local_config",
                "avatar_id": self._schema.avatar_id,
                "avatar_name": self._schema.avatar_name,
                "path": str(self._schema.source_path) if self._schema.source_path else None,
                "parameter_count": len(self._schema.parameters),
            }

        return ok(
            {
                "transport": self._settings.mcp.transport,
                "osc_target": {
                    "ip": self._settings.osc.send.ip,
                    "port": self._settings.osc.send.port,
                },
                "receiver": {
                    "enabled": self._settings.osc.receive.enabled,
                    "ip": self._settings.osc.receive.ip,
                    "port": self._settings.osc.receive.port,
                },
                "queue_depth": self._transport.queue_depth(),
                "safety": {
                    "osc_per_second": self._settings.safety.osc_per_second,
                    "chat_per_minute": self._settings.safety.chat_per_minute,
                    "max_axis_duration_ms": self._settings.safety.max_axis_duration_ms,
                    "max_button_hold_ms": self._settings.safety.max_button_hold_ms,
                    "parameter_policy": self._settings.safety.parameter_policy,
                    "allowed_parameters": list(self._settings.safety.allowed_parameters),
                },
                "schema": schema_info,
            }
        )

    async def set_parameter(self, *, name: str, value: bool | int | float, ttl_ms: int = 0, trace_id: str) -> dict[str, Any]:
        policy = self._settings.safety.parameter_policy
        allowed = set(self._settings.safety.allowed_parameters)

        schema_param = self._schema.resolve(name) if self._schema is not None else None

        if policy == "strict":
            if self._schema is None:
                raise DomainError(
                    code="E_PARAMETER_UNKNOWN",
                    message="未加载 Avatar schema（LocalLow OSC config）。请提供 --avatar-config 或配置 vrchat.avatar_config / vrchat.osc_root。",
                )
            if schema_param is None:
                raise DomainError(
                    code="E_PARAMETER_UNKNOWN",
                    message="参数名不存在于当前 Avatar schema。建议先调用 vrc_list_parameters。",
                    details={"name": name},
                )
            if not schema_param.writable:
                raise DomainError(
                    code="E_SAFETY_REJECTED",
                    message="该参数在 schema 中没有 input address，可能不可写。",
                    details={"name": schema_param.name},
                )

            coerced_value, coerced = self._validate_value(schema_param.type, value)
            address = schema_param.input_address
            assert address is not None

            await self._transport.send(address=address, value=coerced_value, trace_id=trace_id)
            return ok(
                {
                    "trace_id": trace_id,
                    "osc_address": address,
                    "parameter": {"name": schema_param.name, "type": schema_param.type},
                    "sent_value": coerced_value,
                    "value_coerced": coerced,
                    "ttl_ms": ttl_ms,
                }
            )

        if policy == "allowlist" and name not in allowed:
            raise DomainError(
                code="E_SAFETY_REJECTED",
                message="参数不在 allowlist 中，已拒绝发送。",
                details={"name": name, "policy": policy},
            )

        if not isinstance(value, (bool, int, float)):
            raise DomainError(
                code="E_PARAMETER_TYPE_MISMATCH",
                message="value 必须是 bool/int/float。",
                details={"name": name, "value_type": type(value).__name__},
            )

        # If schema is available, prefer its canonical input address and type enforcement.
        address = f"/avatar/parameters/{name}"
        expected_type: ParameterType | None = None
        if schema_param is not None and schema_param.input_address is not None:
            address = schema_param.input_address
            expected_type = schema_param.type

        coerced_value = value
        value_coerced = False
        if expected_type is not None:
            coerced_value, value_coerced = self._validate_value(expected_type, value)

        await self._transport.send(address=address, value=coerced_value, trace_id=trace_id)

        return ok(
            {
                "trace_id": trace_id,
                "osc_address": address,
                "sent_value": coerced_value,
                "value_coerced": value_coerced,
                "ttl_ms": ttl_ms,
            }
        )

    async def input_tap(self, *, button: str, hold_ms: int = 80, trace_id: str) -> dict[str, Any]:
        button_name = _resolve_supported_input(raw=button, allowed=_INPUT_BUTTONS, kind="button")
        # Cap hold time (truncate instead of rejecting)
        max_hold = self._settings.safety.max_button_hold_ms
        effective_hold = max(0, min(int(hold_ms), max_hold))
        capped = effective_hold != int(hold_ms)

        address = f"/input/{button_name}"
        start = time.monotonic()

        try:
            await self._transport.send(address=address, value=1, trace_id=trace_id)
            await asyncio.sleep(effective_hold / 1000)
        finally:
            # Non-negotiable: must release even if cancelled
            await asyncio.shield(self._transport.send(address=address, value=0, trace_id=trace_id))

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ok(
            {
                "trace_id": trace_id,
                "osc_address": address,
                "hold_ms": effective_hold,
                "hold_ms_capped": capped,
                "elapsed_ms": elapsed_ms,
            }
        )

    async def input_axis(self, *, axis: str, value: float, duration_ms: int, trace_id: str) -> dict[str, Any]:
        axis_name = _resolve_supported_input(raw=axis, allowed=_INPUT_AXES, kind="axis")
        # Required + capped duration
        max_dur = self._settings.safety.max_axis_duration_ms
        effective_dur = max(0, min(int(duration_ms), max_dur))
        dur_capped = effective_dur != int(duration_ms)

        v, clamped = _clamp_axis(float(value))
        address = f"/input/{axis_name}"
        start = time.monotonic()

        try:
            await self._transport.send(address=address, value=v, trace_id=trace_id)
            await asyncio.sleep(effective_dur / 1000)
        finally:
            await asyncio.shield(self._transport.send(address=address, value=0.0, trace_id=trace_id))

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ok(
            {
                "trace_id": trace_id,
                "osc_address": address,
                "sent_value": v,
                "value_clamped": clamped,
                "duration_ms": effective_dur,
                "duration_ms_capped": dur_capped,
                "elapsed_ms": elapsed_ms,
            }
        )

    async def chat_send(self, *, text: str, immediate: bool = True, notify: bool = True, trace_id: str) -> dict[str, Any]:
        decision = self._chat_limiter.check()
        if not decision.allowed:
            raise DomainError(
                code="E_RATE_LIMITED",
                message="Chatbox 发送过于频繁，已限流。",
                retry_after_ms=decision.retry_after_ms,
            )

        trimmed = trim_chatbox_text(text)
        address = "/chatbox/input"
        # VRChat expects: (text: str, immediate: bool, notify: bool)
        await self._transport.send(address=address, value=[trimmed, bool(immediate), bool(notify)], trace_id=trace_id)

        return ok(
            {
                "trace_id": trace_id,
                "osc_address": address,
                "trimmed": trimmed,
                "trimmed_len": len(trimmed),
                "immediate": bool(immediate),
                "notify": bool(notify),
            }
        )

    async def chat_typing(self, *, on: bool, trace_id: str) -> dict[str, Any]:
        address = "/chatbox/typing"
        await self._transport.send(address=address, value=bool(on), trace_id=trace_id)
        return ok({"trace_id": trace_id, "osc_address": address, "on": bool(on)})

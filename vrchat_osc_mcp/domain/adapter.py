from __future__ import annotations

import asyncio
import time
from typing import Any

from .chatbox import trim_chatbox_text
from .errors import DomainError
from .safety import SlidingWindowRateLimiter
from .streams import EyeStream, TrackingStream, StreamStatus, tracker_to_osc_index, validate_gaze_mode
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
            code="INVALID_ARGUMENT",
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
        code="INVALID_ARGUMENT",
        message=(
            f"未知/不支持的 VRChat /input {kind}: {raw!r}。"
            "请使用 vrc_input_list_endpoints 查看支持清单，或按 VRChat 官方输入名传参。"
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
        self._held_buttons: set[str] = set()
        self._chat_limiter = SlidingWindowRateLimiter(
            max_events=settings.safety.chat_per_minute,
            window_s=60.0,
        )

        # Background streams (tracking/eye)
        self._tracking_stream = TrackingStream(transport=transport, logger=logger.bind(component="tracking-stream"))
        self._eye_stream = EyeStream(transport=transport, logger=logger.bind(component="eye-stream"))

    # -----------------
    # Meta domain
    # -----------------

    def meta_get_status(self) -> dict[str, Any]:
        tracking = self._tracking_stream.status()
        eye = self._eye_stream.status()

        # We cannot truly probe VRChat OSC readiness (UDP is fire-and-forget).
        # Best-effort signal: if we were able to load LocalLow OSC config, OSC
        # was enabled at least once.
        osc_enabled_detected = self._schema is not None

        def _stream_obj(s: StreamStatus) -> dict[str, Any]:
            d: dict[str, Any] = {
                "running": bool(s.running),
                "stream_id": s.stream_id,
                "fps": s.fps,
            }
            if s.mode is not None:
                d["mode"] = s.mode
            return d

        return {
            "osc_enabled_detected": osc_enabled_detected,
            "target_host": self._settings.osc.send.ip,
            "target_port": self._settings.osc.send.port,
            "last_send_ms_ago": self._transport.last_sent_ms_ago(),
            "streams": {
                "tracking": _stream_obj(tracking),
                "eye": _stream_obj(eye),
            },
        }

    def meta_get_capabilities(self, *, refresh: bool = False) -> dict[str, Any]:
        # refresh 预留给未来：从 /avatar/change 或 receiver 刷新 schema/能力缓存。
        _ = refresh
        return {
            "input_axes_supported": sorted(_INPUT_AXES.keys()),
            "input_buttons_supported": sorted(_INPUT_BUTTONS.keys()),
            "tracking_supported": True,
            "eye_tracking_supported": True,
            "chatbox_supported": True,
            "avatar_parameters_supported": True,
            "notes": {
                "lookhorizontal_vr": "VR 舒适转向开启时 LookHorizontal=1 可能触发 snap-turn（见 VRChat 文档）。",
                "input_write_only": "VRChat /input 是 write-only；不归零会持续生效。",
            },
        }

    # -----------------
    # Avatar domain
    # -----------------

    def avatar_list_parameters(self) -> dict[str, Any]:
        if self._schema is None:
            return {"parameters": []}

        params: list[dict[str, Any]] = []
        for p in self._schema.parameters.values():
            params.append(
                {
                    "name": p.name,
                    "type": p.type,
                    # These fields are not available from LocalLow schema today.
                    "default": None,
                    "min": None,
                    "max": None,
                    "is_network_synced": None,
                }
            )

        params.sort(key=lambda x: x["name"])
        return {"parameters": params}

    async def avatar_set_parameter(
        self,
        *,
        name: str,
        value: Any,
        duration_ms: int = 0,
        reset_value: Any = None,
        trace_id: str,
    ) -> dict[str, Any]:
        # Reuse existing safety policy behavior, but with v1.0 error codes.
        policy = self._settings.safety.parameter_policy
        allowed = set(self._settings.safety.allowed_parameters)

        schema_param = self._schema.resolve(name) if self._schema is not None else None

        if policy == "strict":
            if self._schema is None:
                raise DomainError(
                    code="CAPABILITY_UNAVAILABLE",
                    message="未加载 Avatar schema（LocalLow OSC config）。",
                    details={"hint": "请提供 --avatar-config 或配置 vrchat.avatar_config / vrchat.osc_root。"},
                )
            if schema_param is None:
                raise DomainError(
                    code="INVALID_ARGUMENT",
                    message="参数名不存在于当前 Avatar schema。建议先调用 vrc_avatar_list_parameters。",
                    details={"name": name},
                )
            if not schema_param.writable:
                raise DomainError(
                    code="CAPABILITY_UNAVAILABLE",
                    message="该参数在 schema 中没有 input address，可能不可写。",
                    details={"name": schema_param.name},
                )

            coerced_value, coerced = self._validate_value(schema_param.type, value)
            address = schema_param.input_address
            assert address is not None

            await self._transport.send(address=address, value=coerced_value, trace_id=trace_id)

            if duration_ms and duration_ms > 0:
                await asyncio.sleep(max(0, int(duration_ms)) / 1000)
                rv = reset_value
                if rv is None:
                    rv = False if schema_param.type == "Bool" else (0 if schema_param.type == "Int" else 0.0)
                rv2, _ = self._validate_value(schema_param.type, rv)
                await asyncio.shield(self._transport.send(address=address, value=rv2, trace_id=trace_id))

            return {
                "osc_address": address,
                "sent_value": coerced_value,
                "value_coerced": coerced,
                "duration_ms": int(duration_ms) if duration_ms else 0,
            }

        if policy == "allowlist" and name not in allowed:
            raise DomainError(
                code="CAPABILITY_UNAVAILABLE",
                message="参数不在 allowlist 中，已拒绝发送。",
                details={"name": name, "policy": policy},
            )

        if not isinstance(value, (bool, int, float)):
            raise DomainError(
                code="INVALID_ARGUMENT",
                message="value 必须是 bool/int/float。",
                details={"name": name, "value_type": type(value).__name__},
            )

        address = f"/avatar/parameters/{name}"
        expected_type: ParameterType | None = None
        if schema_param is not None and schema_param.input_address is not None:
            address = schema_param.input_address
            expected_type = schema_param.type

        coerced_value: bool | int | float = value
        value_coerced = False
        if expected_type is not None:
            coerced_value, value_coerced = self._validate_value(expected_type, value)

        await self._transport.send(address=address, value=coerced_value, trace_id=trace_id)

        if duration_ms and duration_ms > 0:
            await asyncio.sleep(max(0, int(duration_ms)) / 1000)
            rv = reset_value
            if rv is None:
                # Best-effort default reset
                rv = False if isinstance(coerced_value, bool) else 0.0
            if not isinstance(rv, (bool, int, float)):
                raise DomainError(
                    code="INVALID_ARGUMENT",
                    message="reset_value 必须是 bool/int/float。",
                    details={"name": name, "reset_value_type": type(rv).__name__},
                )
            await asyncio.shield(self._transport.send(address=address, value=rv, trace_id=trace_id))

        return {
            "osc_address": address,
            "sent_value": coerced_value,
            "value_coerced": value_coerced,
            "duration_ms": int(duration_ms) if duration_ms else 0,
        }

    async def avatar_set_parameters(self, *, parameters: list[dict[str, Any]], trace_id: str) -> dict[str, Any]:
        if not parameters:
            raise DomainError(code="INVALID_ARGUMENT", message="parameters 不能为空。")

        results: list[dict[str, Any]] = []
        for p in parameters:
            if not isinstance(p, dict):
                raise DomainError(code="INVALID_ARGUMENT", message="parameters[*] 必须是对象。")
            name = p.get("name")
            if not isinstance(name, str) or not name:
                raise DomainError(code="INVALID_ARGUMENT", message="parameters[*].name 必须是非空字符串。")
            if "value" not in p:
                raise DomainError(code="INVALID_ARGUMENT", message="parameters[*].value 是必填。", details={"name": name})
            value = p.get("value")
            r = await self.avatar_set_parameter(name=name, value=value, duration_ms=0, reset_value=None, trace_id=trace_id)
            results.append({"name": name, "osc_address": r.get("osc_address"), "sent_value": r.get("sent_value")})

        return {"results": results}

    @staticmethod
    def _validate_value(expected: ParameterType, value: bool | int | float) -> tuple[bool | int | float, bool]:
        """Return (coerced_value, coerced_flag) or raise DomainError."""

        if expected == "Bool":
            if isinstance(value, bool):
                return value, False
            raise DomainError(code="INVALID_ARGUMENT", message="参数类型为 Bool，但 value 不是 bool。")

        if expected == "Int":
            # bool is a subclass of int in Python, so exclude it explicitly.
            if isinstance(value, bool) or not isinstance(value, int):
                raise DomainError(code="INVALID_ARGUMENT", message="参数类型为 Int，但 value 不是 int。")
            return value, False

        # Float
        if isinstance(value, bool):
            raise DomainError(code="INVALID_ARGUMENT", message="参数类型为 Float，但 value 不是 number。")
        if isinstance(value, (int, float)):
            coerced = float(value)
            return coerced, coerced != value

        raise DomainError(code="INVALID_ARGUMENT", message="value 必须是 bool/int/float。")

    # -----------------
    # Input domain (/input)
    # -----------------

    def input_list_endpoints(self) -> dict[str, Any]:
        return {"axes": sorted(_INPUT_AXES.keys()), "buttons": sorted(_INPUT_BUTTONS.keys())}

    async def input_tap_buttons(self, *, buttons: list[str], press_ms: int = 80, trace_id: str) -> dict[str, Any]:
        if not buttons:
            raise DomainError(code="INVALID_ARGUMENT", message="buttons 不能为空。")

        press_ms_i = int(press_ms)
        if press_ms_i < 20:
            press_ms_i = 20
        if press_ms_i > 1000:
            press_ms_i = 1000

        results: list[dict[str, Any]] = []
        for raw in buttons:
            name = _resolve_supported_input(raw=raw, allowed=_INPUT_BUTTONS, kind="button")
            address = f"/input/{name}"
            start = time.monotonic()
            try:
                await self._transport.send(address=address, value=1, trace_id=trace_id)
                await asyncio.sleep(press_ms_i / 1000)
            finally:
                await asyncio.shield(self._transport.send(address=address, value=0, trace_id=trace_id))
            elapsed_ms = int((time.monotonic() - start) * 1000)
            results.append({"button": name, "osc_address": address, "elapsed_ms": elapsed_ms})
        return {"results": results}

    async def input_hold_buttons(self, *, buttons: list[str], trace_id: str) -> dict[str, Any]:
        if not buttons:
            raise DomainError(code="INVALID_ARGUMENT", message="buttons 不能为空。")

        held: list[str] = []
        for raw in buttons:
            name = _resolve_supported_input(raw=raw, allowed=_INPUT_BUTTONS, kind="button")
            address = f"/input/{name}"
            await self._transport.send(address=address, value=1, trace_id=trace_id)
            self._held_buttons.add(name)
            held.append(name)
        return {"held": held}

    async def input_release_buttons(self, *, buttons: list[str], trace_id: str) -> dict[str, Any]:
        if not buttons:
            raise DomainError(code="INVALID_ARGUMENT", message="buttons 不能为空。")

        released: list[str] = []
        for raw in buttons:
            name = _resolve_supported_input(raw=raw, allowed=_INPUT_BUTTONS, kind="button")
            address = f"/input/{name}"
            await self._transport.send(address=address, value=0, trace_id=trace_id)
            self._held_buttons.discard(name)
            released.append(name)
        return {"released": released}

    async def input_set_axes(
        self,
        *,
        axes: dict[str, Any],
        duration_ms: int = 0,
        auto_zero: bool = True,
        ease_ms: int = 80,
        trace_id: str,
    ) -> dict[str, Any]:
        if not isinstance(axes, dict):
            raise DomainError(code="INVALID_ARGUMENT", message="axes 必须是对象。")

        allowed_axes = {"Vertical", "Horizontal", "LookHorizontal"}
        unknown = [k for k in axes.keys() if k not in allowed_axes]
        if unknown:
            raise DomainError(
                code="INVALID_ARGUMENT",
                message="axes 包含不支持的字段。",
                details={"unknown": sorted(unknown), "supported": sorted(allowed_axes)},
            )

        # Normalize and clamp
        to_send: dict[str, float] = {}
        for k, v in axes.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise DomainError(code="INVALID_ARGUMENT", message="axis 值必须是 number。", details={"axis": k, "value_type": type(v).__name__})
            vv, _clamped = _clamp_axis(float(v))
            to_send[k] = vv

        if not to_send:
            raise DomainError(code="INVALID_ARGUMENT", message="axes 不能为空对象。")

        # duration safety: if auto_zero, enforce bounded duration (default to ease_ms)
        effective_dur = int(duration_ms) if duration_ms is not None else 0
        if auto_zero and effective_dur <= 0:
            effective_dur = int(ease_ms) if ease_ms is not None else 80

        max_dur = int(self._settings.safety.max_axis_duration_ms)
        if effective_dur > max_dur:
            effective_dur = max_dur

        start = time.monotonic()
        try:
            for axis_name, vv in to_send.items():
                await self._transport.send(address=f"/input/{axis_name}", value=vv, trace_id=trace_id)
            if auto_zero:
                await asyncio.sleep(max(0, effective_dur) / 1000)
        finally:
            if auto_zero:
                for axis_name in to_send.keys():
                    await asyncio.shield(self._transport.send(address=f"/input/{axis_name}", value=0.0, trace_id=trace_id))

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {
            "axes": to_send,
            "duration_ms": effective_dur if auto_zero else 0,
            "auto_zero": bool(auto_zero),
            "ease_ms": int(ease_ms),
            "elapsed_ms": elapsed_ms,
        }

    async def input_stop(self, *, trace_id: str) -> dict[str, Any]:
        # Emergency stop: release everything we know.
        for axis_name in ["Vertical", "Horizontal", "LookHorizontal"]:
            await self._transport.send(address=f"/input/{axis_name}", value=0.0, trace_id=trace_id)

        # Release all known buttons (safe, idempotent)
        for btn in sorted(_INPUT_BUTTONS.keys()):
            await self._transport.send(address=f"/input/{btn}", value=0, trace_id=trace_id)
        self._held_buttons.clear()
        return {"stopped": True}

    # -----------------
    # Chatbox domain
    # -----------------

    async def chatbox_send(
        self,
        *,
        text: str,
        send_immediately: bool = True,
        notify: bool = True,
        set_typing: bool = False,
        trace_id: str,
    ) -> dict[str, Any]:
        decision = self._chat_limiter.check()
        if not decision.allowed:
            raise DomainError(
                code="RATE_LIMITED",
                message="Chatbox 发送过于频繁，已限流。",
                details={"retry_after_ms": decision.retry_after_ms},
            )

        trimmed = trim_chatbox_text(text)

        if set_typing:
            await self.chatbox_set_typing(is_typing=True, trace_id=trace_id)

        address = "/chatbox/input"
        # VRChat expects: (text: str, immediate: bool, notify: bool)
        await self._transport.send(
            address=address,
            value=[trimmed, bool(send_immediately), bool(notify)],
            trace_id=trace_id,
        )

        # If we bypass keyboard and immediately send, we can clear typing.
        if set_typing and send_immediately:
            await self.chatbox_set_typing(is_typing=False, trace_id=trace_id)

        return {
            "osc_address": address,
            "trimmed": trimmed,
            "trimmed_len": len(trimmed),
            "send_immediately": bool(send_immediately),
            "notify": bool(notify),
        }

    async def chatbox_set_typing(self, *, is_typing: bool, trace_id: str) -> dict[str, Any]:
        address = "/chatbox/typing"
        await self._transport.send(address=address, value=bool(is_typing), trace_id=trace_id)
        return {"osc_address": address, "is_typing": bool(is_typing)}

    # -----------------
    # Tracking domain
    # -----------------

    def tracking_status(self) -> dict[str, Any]:
        s = self._tracking_stream.status()
        return {"running": s.running, "stream_id": s.stream_id, "fps": s.fps}

    async def tracking_set_tracker_pose(
        self,
        *,
        tracker: str,
        position_m: dict[str, Any],
        rotation_euler_deg: dict[str, Any],
        trace_id: str,
    ) -> dict[str, Any]:
        idx = tracker_to_osc_index(tracker)
        try:
            pos = (float(position_m["x"]), float(position_m["y"]), float(position_m["z"]))
            rot = (float(rotation_euler_deg["x"]), float(rotation_euler_deg["y"]), float(rotation_euler_deg["z"]))
        except Exception as e:  # noqa: BLE001
            raise DomainError(code="INVALID_ARGUMENT", message="position_m/rotation_euler_deg 必须包含 x,y,z 数字字段。") from e

        self._tracking_stream.set_tracker_pose(tracker=tracker, position_m=pos, rotation_euler_deg=rot)
        return {
            "tracker": tracker,
            "osc_index": idx,
            "position_m": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "rotation_euler_deg": {"x": rot[0], "y": rot[1], "z": rot[2]},
        }

    async def tracking_set_head_reference(
        self,
        *,
        position_m: dict[str, Any] | None,
        rotation_euler_deg: dict[str, Any] | None,
        mode: str,
        trace_id: str,
    ) -> dict[str, Any]:
        if mode not in {"single_align", "stream_align"}:
            raise DomainError(code="INVALID_ARGUMENT", message="mode 必须是 single_align 或 stream_align。")

        pos_t: tuple[float, float, float] | None = None
        rot_t: tuple[float, float, float] | None = None

        if position_m is not None:
            try:
                pos_t = (float(position_m["x"]), float(position_m["y"]), float(position_m["z"]))
            except Exception as e:  # noqa: BLE001
                raise DomainError(code="INVALID_ARGUMENT", message="position_m 必须包含 x,y,z 数字字段。") from e

        if rotation_euler_deg is not None:
            try:
                rot_t = (float(rotation_euler_deg["x"]), float(rotation_euler_deg["y"]), float(rotation_euler_deg["z"]))
            except Exception as e:  # noqa: BLE001
                raise DomainError(code="INVALID_ARGUMENT", message="rotation_euler_deg 必须包含 x,y,z 数字字段。") from e

        await self._tracking_stream.set_head_reference(
            position_m=pos_t,
            rotation_euler_deg=rot_t,
            mode=mode,  # type: ignore[arg-type]
            trace_id=trace_id,
        )

        return {"mode": mode, "position_m": position_m, "rotation_euler_deg": rotation_euler_deg}

    async def tracking_stream_start(
        self,
        *,
        fps: int,
        enabled_trackers: list[str],
        neutral_on_stop: bool,
        trace_id: str,
    ) -> dict[str, Any]:
        # Validate tracker names early
        for t in enabled_trackers:
            tracker_to_osc_index(t)
        stream_id = await self._tracking_stream.start(fps=fps, enabled_trackers=enabled_trackers, neutral_on_stop=neutral_on_stop)
        return {"running": True, "stream_id": stream_id, "fps": fps}

    async def tracking_stream_stop(self, *, stream_id: str | None, trace_id: str) -> dict[str, Any]:
        _ = stream_id
        await self._tracking_stream.stop(trace_id=trace_id)
        return {"stopped": True}

    # -----------------
    # Eye domain
    # -----------------

    async def eye_stream_start(self, *, fps: int, gaze_mode: str, neutral_on_stop: bool, trace_id: str) -> dict[str, Any]:
        gaze_mode = validate_gaze_mode(gaze_mode)
        stream_id = await self._eye_stream.start(fps=fps, gaze_mode=gaze_mode, neutral_on_stop=neutral_on_stop)
        return {"running": True, "stream_id": stream_id, "fps": fps, "mode": gaze_mode}

    async def eye_stream_stop(self, *, stream_id: str | None, trace_id: str) -> dict[str, Any]:
        _ = stream_id
        await self._eye_stream.stop(trace_id=trace_id)
        return {"stopped": True}

    async def eye_set_blink(self, *, amount: float, trace_id: str) -> dict[str, Any]:
        self._eye_stream.set_blink(amount=float(amount))
        # Send once immediately as well (useful even without stream)
        await self._transport.send(address="/tracking/eye/EyesClosedAmount", value=float(amount), trace_id=trace_id)
        return {"amount": float(amount)}

    async def eye_set_gaze(self, *, gaze_mode: str, data: dict[str, Any], trace_id: str) -> dict[str, Any]:
        gaze_mode = validate_gaze_mode(gaze_mode)
        if not isinstance(data, dict):
            raise DomainError(code="INVALID_ARGUMENT", message="data 必须是对象。")

        # Strict per-mode validation (v1)
        args: list[float]
        try:
            if gaze_mode == "CenterPitchYaw":
                args = [float(data["pitch"]), float(data["yaw"])]
            elif gaze_mode == "CenterPitchYawDist":
                args = [float(data["pitch"]), float(data["yaw"]), float(data["distance_m"])]
            elif gaze_mode in {"CenterVec", "CenterVecFull"}:
                args = [float(data["x"]), float(data["y"]), float(data["z"])]
            elif gaze_mode == "LeftRightPitchYaw":
                args = [
                    float(data["left_pitch"]),
                    float(data["left_yaw"]),
                    float(data["right_pitch"]),
                    float(data["right_yaw"]),
                ]
            else:  # LeftRightVec
                args = [
                    float(data["left_x"]),
                    float(data["left_y"]),
                    float(data["left_z"]),
                    float(data["right_x"]),
                    float(data["right_y"]),
                    float(data["right_z"]),
                ]
        except KeyError as e:
            raise DomainError(
                code="INVALID_ARGUMENT",
                message="data 缺少 gaze_mode 对应的必需字段。",
                details={"gaze_mode": gaze_mode, "missing": str(e)},
            ) from e
        except Exception as e:  # noqa: BLE001
            raise DomainError(code="INVALID_ARGUMENT", message="data 字段必须是 number。", details={"gaze_mode": gaze_mode}) from e

        # Enforce gaze_mode lock when stream is running
        self._eye_stream.set_gaze(gaze_mode=gaze_mode, args=args)

        # Also send once immediately (useful before stream start)
        await self._transport.send(address=f"/tracking/eye/{gaze_mode}", value=args, trace_id=trace_id)
        return {"gaze_mode": gaze_mode, "sent": True}

    # -----------------
    # Global stop
    # -----------------

    async def stop_all(self, *, trace_id: str) -> dict[str, Any]:
        # Best-effort: do not fail if one subsystem errors.
        errors: list[dict[str, Any]] = []

        async def _try(label: str, coro):
            try:
                await coro
            except DomainError as e:
                errors.append({"subsystem": label, "error": e.to_error_obj()})
            except Exception as e:  # noqa: BLE001
                errors.append({"subsystem": label, "error": {"code": "INTERNAL_ERROR", "message": str(e)}})

        await _try("input", self.input_stop(trace_id=trace_id))
        await _try("tracking", self.tracking_stream_stop(stream_id=None, trace_id=trace_id))
        await _try("eye", self.eye_stream_stop(stream_id=None, trace_id=trace_id))
        await _try("chatbox", self.chatbox_set_typing(is_typing=False, trace_id=trace_id))

        return {"stopped": True, "errors": errors}

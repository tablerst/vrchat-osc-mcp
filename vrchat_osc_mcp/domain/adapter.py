from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any

from .chatbox import trim_chatbox_text
from .errors import DomainError
from .safety import SlidingWindowRateLimiter
from .streams import EyeStream, TrackingStream, StreamStatus, tracker_to_osc_index, validate_gaze_mode
from ..vrc_config.parser import load_avatar_schema
from ..vrc_config.resolver import resolve_avatar_config_path_for_avatar_id
from ..vrc_config.schema import AvatarSchema, ParameterType


def _clamp_axis(v: float) -> tuple[float, bool]:
    if v < -1.0:
        return -1.0, True
    if v > 1.0:
        return 1.0, True
    return v, False


_SUPPORTED_INPUT_AXES: dict[str, str] = {
    # v1 per PLAN.md: only expose 3 most commonly used axes (inputSchema includes only these 3)
    # Official doc.md has more /input axes, but we don't claim v1 support to avoid LLM misuse
    "Vertical": "Move forward(1) / backward(-1)",
    "Horizontal": "Move right(1) / left(-1)",
    "LookHorizontal": "Look left/right; desktop mode enables smooth turning; VR comfort-turn snap at value 1",
}

_INPUT_BUTTONS: dict[str, str] = {
    "MoveForward": "Move forward when value is 1",
    "MoveBackward": "Move backward when value is 1",
    "MoveLeft": "Move left when value is 1",
    "MoveRight": "Move right when value is 1",
    "LookLeft": "Turn left at value 1 (desktop smooth; VR comfort-turn snap)",
    "LookRight": "Turn right at value 1 (desktop smooth; VR comfort-turn snap)",
    "Jump": "Jump (effective when world supports jumping)",
    "Run": "Run (effective when world supports running)",
    "ComfortLeft": "Quick turn left (VR only)",
    "ComfortRight": "Quick turn right (VR only)",
    "DropRight": "Drop item held in right hand (VR only)",
    "UseRight": "Use item highlighted by right hand (VR only)",
    "GrabRight": "Grab item highlighted by right hand (VR only)",
    "DropLeft": "Drop item held in left hand (VR only)",
    "UseLeft": "Use item highlighted by left hand (VR only)",
    "GrabLeft": "Grab item highlighted by left hand (VR only)",
    "PanicButton": "Open safety mode",
    "QuickMenuToggleLeft": "Toggle quick menu: 0->1 transition triggers toggle",
    "QuickMenuToggleRight": "Toggle quick menu: 0->1 transition triggers toggle",
    "Voice": "Voice toggle/mute (behavior depends on VRChat's 'Toggle Voice' setting)",
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
            message=f"{kind} cannot be empty.",
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
            f"Unknown/unsupported VRChat /input {kind}: {raw!r}. "
            "Use vrc_input_list_endpoints to view supported list, or pass VRChat official input name."
        ),
        details={
            "kind": kind,
            "given": raw,
            "normalized": name,
            "supported": sorted(allowed.keys()),
        },
    )


@dataclass(frozen=True)
class _SchemaState:
    """Immutable snapshot of avatar schema + refresh status.

    We keep the last-known-good schema even if refresh fails.
    """

    # Last observed /avatar/change value (authoritative when present)
    observed_avatar_id: str | None

    # Last-known-good schema
    schema: AvatarSchema | None
    schema_source: str | None
    schema_path: str | None
    schema_loaded_at_ms: int | None

    # Refresh attempt tracking
    schema_last_refresh_attempt_at_ms: int | None
    schema_last_refresh_ok_at_ms: int | None
    schema_last_error: dict[str, Any] | None

    @classmethod
    def initial(cls, *, schema: AvatarSchema | None, schema_source: str | None, schema_path: str | None) -> "_SchemaState":
        now_ms = int(time.time() * 1000)
        loaded_at = now_ms if schema is not None else None
        return cls(
            observed_avatar_id=None,
            schema=schema,
            schema_source=schema_source,
            schema_path=schema_path,
            schema_loaded_at_ms=loaded_at,
            schema_last_refresh_attempt_at_ms=None,
            schema_last_refresh_ok_at_ms=loaded_at,
            schema_last_error=None,
        )

    def _with(self, **kwargs: Any) -> "_SchemaState":
        return replace(self, **kwargs)


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
        schema_source: str | None = None,
        schema_path: str | None = None,
    ) -> None:
        self._transport = transport
        self._settings = settings
        self._logger = logger
        self._schema_state = _SchemaState.initial(
            schema=schema,
            schema_source=schema_source,
            schema_path=schema_path,
        )
        self._held_buttons: set[str] = set()
        self._chat_limiter = SlidingWindowRateLimiter(
            max_events=settings.safety.chat_per_minute,
            window_s=60.0,
        )

        # Background streams (tracking/eye)
        self._tracking_stream = TrackingStream(
            transport=transport,
            logger=logger.bind(component="tracking-stream"),
            target_ttl_ms=int(settings.safety.tracking_target_ttl_ms),
        )
        self._eye_stream = EyeStream(
            transport=transport,
            logger=logger.bind(component="eye-stream"),
            target_ttl_ms=int(settings.safety.eye_target_ttl_ms),
        )

    def _schema_snapshot(self) -> "_SchemaState":
        # Copy-on-read snapshot: this reference is treated as immutable.
        return self._schema_state

    async def on_avatar_change(self, avatar_id: str) -> None:
        """Handle /avatar/change.

        Contract:
        - Always record the observed avatar_id.
        - If schema refresh fails, keep the last-known-good schema.
        - strict policy continues to validate against the last-known-good schema.
        """

        avatar_id = (avatar_id or "").strip()
        if not avatar_id:
            return

        now_ms = int(time.time() * 1000)
        old = self._schema_snapshot()

        # Record attempt immediately (even before we know if we can load schema).
        self._schema_state = old._with(
            observed_avatar_id=avatar_id,
            schema_last_refresh_attempt_at_ms=now_ms,
        )

        schema_path = resolve_avatar_config_path_for_avatar_id(
            osc_root=self._settings.vrchat.osc_root,
            avatar_id=avatar_id,
        )

        if schema_path is None:
            err = {
                "code": "CAPABILITY_UNAVAILABLE",
                "message": "Avatar LocalLow OSC config file not found (may be Build & Test or config not generated).",
                "details": {
                    "avatar_id": avatar_id,
                    "hint": "If this Avatar was only loaded via Build & Test, VRChat does not save OSC config to disk.",
                },
            }
            self._schema_state = self._schema_snapshot()._with(
                schema_last_error=err,
            )
            self._logger.warning(
                "schema.refresh_failed",
                schema_source="local_config_by_avatar_id",
                avatar_id=avatar_id,
                reason="config_not_found",
            )
            return

        try:
            new_schema = load_avatar_schema(schema_path)
        except Exception as e:  # noqa: BLE001
            err = {
                "code": "CAPABILITY_UNAVAILABLE",
                "message": "Failed to read/parse Avatar OSC config, keeping old schema.",
                "details": {
                    "avatar_id": avatar_id,
                    "schema_path": str(schema_path),
                    "error": str(e),
                },
            }
            self._schema_state = self._schema_snapshot()._with(
                schema_last_error=err,
            )
            self._logger.warning(
                "schema.refresh_failed",
                schema_source="local_config_by_avatar_id",
                avatar_id=avatar_id,
                schema_path=str(schema_path),
                reason="load_failed",
                error=str(e),
            )
            return

        # Success: swap schema snapshot.
        self._schema_state = self._schema_snapshot()._with(
            schema=new_schema,
            schema_source="local_config_by_avatar_id",
            schema_path=str(schema_path),
            schema_loaded_at_ms=now_ms,
            schema_last_refresh_ok_at_ms=now_ms,
            schema_last_error=None,
        )
        self._logger.info(
            "schema.refreshed",
            schema_source="local_config_by_avatar_id",
            avatar_id=new_schema.avatar_id,
            schema_path=str(schema_path),
        )

    # -----------------
    # Meta domain
    # -----------------

    def meta_get_status(self) -> dict[str, Any]:
        tracking = self._tracking_stream.status()
        eye = self._eye_stream.status()

        # We cannot truly probe VRChat OSC readiness (UDP is fire-and-forget).
        # Best-effort signal: if we were able to load LocalLow OSC config, OSC
        # was enabled at least once.
        osc_enabled_detected = self._schema_snapshot().schema is not None

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
        # refresh reserved for future: refresh schema/capability cache from /avatar/change or receiver
        _ = refresh

        s = self._schema_snapshot()
        schema_avatar_id = s.schema.avatar_id if s.schema is not None else None
        schema_stale = bool(s.observed_avatar_id and schema_avatar_id and s.observed_avatar_id != schema_avatar_id)
        if s.schema is None:
            schema_confidence = "none"
        elif s.observed_avatar_id is None:
            schema_confidence = "guessed"
        elif schema_stale:
            schema_confidence = "stale"
        else:
            schema_confidence = "observed"

        return {
            "input_axes_supported": sorted(_SUPPORTED_INPUT_AXES.keys()),
            "input_buttons_supported": sorted(_INPUT_BUTTONS.keys()),
            "tracking_supported": True,
            "eye_tracking_supported": True,
            "chatbox_supported": True,
            "avatar_parameters_supported": True,
            "avatar": {
                "current_avatar_id": s.observed_avatar_id,
                "schema_avatar_id": schema_avatar_id,
                "schema_source": s.schema_source,
                "schema_path": s.schema_path,
                "schema_confidence": schema_confidence,
                "schema_stale": schema_stale,
                "schema_loaded_at_ms": s.schema_loaded_at_ms,
                "schema_last_refresh_attempt_at_ms": s.schema_last_refresh_attempt_at_ms,
                "schema_last_refresh_ok_at_ms": s.schema_last_refresh_ok_at_ms,
                "schema_last_error": s.schema_last_error,
            },
            "notes": {
                "lookhorizontal_vr": "LookHorizontal=1 may trigger snap-turn when VR comfort-turn is enabled (see VRChat docs).",
                "input_write_only": "VRChat /input is write-only; not resetting causes persistent effect.",
            },
        }

    # -----------------
    # Avatar domain
    # -----------------

    def avatar_list_parameters(self) -> dict[str, Any]:
        schema = self._schema_snapshot().schema
        if schema is None:
            return {"parameters": []}

        params: list[dict[str, Any]] = []
        for p in schema.parameters.values():
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
        if not isinstance(name, str) or not name.strip():
            raise DomainError(code="INVALID_ARGUMENT", message="name must be non-empty string.")

        # Reuse existing safety policy behavior, but with v1.0 error codes.
        policy = self._settings.safety.parameter_policy
        allowed = set(self._settings.safety.allowed_parameters)

        schema = self._schema_snapshot().schema
        schema_param = schema.resolve(name) if schema is not None else None

        if policy == "strict":
            if schema is None:
                raise DomainError(
                    code="CAPABILITY_UNAVAILABLE",
                    message="Avatar schema (LocalLow OSC config) not loaded.",
                    details={"hint": "Provide --avatar-config or configure vrchat.avatar_config / vrchat.osc_root."},
                )
            if schema_param is None:
                raise DomainError(
                    code="INVALID_ARGUMENT",
                    message="Parameter name does not exist in current Avatar schema. Recommend calling vrc_avatar_list_parameters first.",
                    details={"name": name},
                )
            if not schema_param.writable:
                raise DomainError(
                    code="CAPABILITY_UNAVAILABLE",
                    message="Parameter has no input address in schema, may not be writable.",
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
                message="Parameter not in allowlist, send rejected.",
                details={"name": name, "policy": policy},
            )

        if not isinstance(value, (bool, int, float)):
            raise DomainError(
                code="INVALID_ARGUMENT",
                message="value must be bool/int/float.",
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
                    message="reset_value must be bool/int/float.",
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
            raise DomainError(code="INVALID_ARGUMENT", message="parameters cannot be empty.")

        results: list[dict[str, Any]] = []
        for p in parameters:
            if not isinstance(p, dict):
                raise DomainError(code="INVALID_ARGUMENT", message="parameters[*] must be object.")
            name = p.get("name")
            if not isinstance(name, str) or not name:
                raise DomainError(code="INVALID_ARGUMENT", message="parameters[*].name must be non-empty string.")
            if "value" not in p:
                raise DomainError(code="INVALID_ARGUMENT", message="parameters[*].value is required.", details={"name": name})
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
            raise DomainError(code="INVALID_ARGUMENT", message="Parameter type is Bool, but value is not bool.")

        if expected == "Int":
            # bool is a subclass of int in Python, so exclude it explicitly.
            if isinstance(value, bool) or not isinstance(value, int):
                raise DomainError(code="INVALID_ARGUMENT", message="Parameter type is Int, but value is not int.")
            return value, False

        # Float
        if isinstance(value, bool):
            raise DomainError(code="INVALID_ARGUMENT", message="Parameter type is Float, but value is not number.")
        if isinstance(value, (int, float)):
            coerced = float(value)
            return coerced, coerced != value

        raise DomainError(code="INVALID_ARGUMENT", message="value must be bool/int/float.")

    # -----------------
    # Input domain (/input)
    # -----------------

    def input_list_endpoints(self) -> dict[str, Any]:
        return {"axes": sorted(_SUPPORTED_INPUT_AXES.keys()), "buttons": sorted(_INPUT_BUTTONS.keys())}

    async def input_tap_buttons(self, *, buttons: list[str], press_ms: int = 80, trace_id: str) -> dict[str, Any]:
        if not buttons:
            raise DomainError(code="INVALID_ARGUMENT", message="buttons cannot be empty.")

        if len(buttons) > 10:
            raise DomainError(code="INVALID_ARGUMENT", message="too many buttons (max 10).", details={"max": 10, "given": len(buttons)})

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
            raise DomainError(code="INVALID_ARGUMENT", message="buttons cannot be empty.")

        if len(buttons) > 10:
            raise DomainError(code="INVALID_ARGUMENT", message="too many buttons (max 10).", details={"max": 10, "given": len(buttons)})

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
            raise DomainError(code="INVALID_ARGUMENT", message="buttons cannot be empty.")

        if len(buttons) > 10:
            raise DomainError(code="INVALID_ARGUMENT", message="too many buttons (max 10).", details={"max": 10, "given": len(buttons)})

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
            raise DomainError(code="INVALID_ARGUMENT", message="axes must be an object.")

        allowed_axes = set(_SUPPORTED_INPUT_AXES.keys())
        unknown = [k for k in axes.keys() if k not in allowed_axes]
        if unknown:
            raise DomainError(
                code="INVALID_ARGUMENT",
                message="axes contains unsupported fields.",
                details={"unknown": sorted(unknown), "supported": sorted(allowed_axes)},
            )

        if duration_ms is None:
            duration_ms = 0
        if not isinstance(duration_ms, int):
            raise DomainError(code="INVALID_ARGUMENT", message="duration_ms must be an integer.")
        if duration_ms < 0:
            raise DomainError(code="INVALID_ARGUMENT", message="duration_ms cannot be negative.", details={"duration_ms": duration_ms})

        if ease_ms is None:
            ease_ms = 80
        if not isinstance(ease_ms, int):
            raise DomainError(code="INVALID_ARGUMENT", message="ease_ms must be an integer.")
        if ease_ms < 0 or ease_ms > 500:
            raise DomainError(code="INVALID_ARGUMENT", message="ease_ms must be within [0, 500].", details={"ease_ms": ease_ms})

        # Normalize and clamp
        to_send: dict[str, float] = {}
        for k, v in axes.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise DomainError(code="INVALID_ARGUMENT", message="axis values must be numbers.", details={"axis": k, "value_type": type(v).__name__})
            vv, _clamped = _clamp_axis(float(v))
            to_send[k] = vv

        if not to_send:
            raise DomainError(code="INVALID_ARGUMENT", message="axes cannot be an empty object.")

        # Safety valve (doc.md): axes should reset to 0 when not in use.
        # To avoid a "perpetual-motion" forward movement, even if the caller passes auto_zero=false,
        # we still apply a bounded forced reset-to-zero.
        requested_auto_zero = bool(auto_zero)
        enforced_auto_zero = True

        # duration safety: enforce bounded duration (default to ease_ms)
        effective_dur = int(duration_ms)
        if effective_dur <= 0:
            effective_dur = int(ease_ms)

        max_dur = int(self._settings.safety.max_axis_duration_ms)
        if effective_dur > max_dur:
            effective_dur = max_dur

        start = time.monotonic()
        try:
            # Prefer a single OSC bundle for multi-axis updates.
            # Rationale: reduce UDP packet count and keep axes temporally aligned.
            items = [(f"/input/{axis_name}", vv) for axis_name, vv in to_send.items()]
            if len(items) > 1 and hasattr(self._transport, "send_bundle"):
                await self._transport.send_bundle(items=items, trace_id=trace_id)
            else:
                for address, vv in items:
                    await self._transport.send(address=address, value=vv, trace_id=trace_id)
            if enforced_auto_zero:
                await asyncio.sleep(max(0, effective_dur) / 1000)
        finally:
            if enforced_auto_zero:
                reset_items = [(f"/input/{axis_name}", 0.0) for axis_name in to_send.keys()]
                if len(reset_items) > 1 and hasattr(self._transport, "send_bundle"):
                    await asyncio.shield(self._transport.send_bundle(items=reset_items, trace_id=trace_id))
                else:
                    for address, vv in reset_items:
                        await asyncio.shield(self._transport.send(address=address, value=vv, trace_id=trace_id))

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {
            "axes": to_send,
            "duration_ms": effective_dur,
            "requested_auto_zero": requested_auto_zero,
            "enforced_auto_zero": enforced_auto_zero,
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
        if not isinstance(text, str) or not text.strip():
            raise DomainError(code="INVALID_ARGUMENT", message="text must be a non-empty string.")

        decision = self._chat_limiter.check()
        if not decision.allowed:
            raise DomainError(
                code="RATE_LIMITED",
                message="Chatbox messages are being sent too frequently; rate limited.",
                details={"retry_after_ms": decision.retry_after_ms},
            )

        trimmed = trim_chatbox_text(text)

        if set_typing:
            await self.chatbox_set_typing(is_typing=True, trace_id=trace_id)

        address = "/chatbox/input"
        # doc.md: /chatbox/input s b n
        # - b=True: send immediately (bypass keyboard)
        # - b=False: open the keyboard and prefill the text
        # - n optional; only used to control notification sound when sending immediately
        if bool(send_immediately):
            value = [trimmed, True, bool(notify)]
        else:
            value = [trimmed, False]

        await self._transport.send(address=address, value=value, trace_id=trace_id)

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
            raise DomainError(code="INVALID_ARGUMENT", message="position_m/rotation_euler_deg must contain numeric x, y, z fields.") from e

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
            raise DomainError(code="INVALID_ARGUMENT", message="mode must be 'single_align' or 'stream_align'.")

        pos_t: tuple[float, float, float] | None = None
        rot_t: tuple[float, float, float] | None = None

        if position_m is not None:
            try:
                pos_t = (float(position_m["x"]), float(position_m["y"]), float(position_m["z"]))
            except Exception as e:  # noqa: BLE001
                raise DomainError(code="INVALID_ARGUMENT", message="position_m must contain numeric x, y, z fields.") from e

        if rotation_euler_deg is not None:
            try:
                rot_t = (float(rotation_euler_deg["x"]), float(rotation_euler_deg["y"]), float(rotation_euler_deg["z"]))
            except Exception as e:  # noqa: BLE001
                raise DomainError(code="INVALID_ARGUMENT", message="rotation_euler_deg must contain numeric x, y, z fields.") from e

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
        if fps not in {20, 30, 45, 60}:
            raise DomainError(code="INVALID_ARGUMENT", message="fps must be one of 20/30/45/60.", details={"fps": fps})
        if not isinstance(enabled_trackers, list) or not enabled_trackers:
            raise DomainError(code="INVALID_ARGUMENT", message="enabled_trackers cannot be empty.")
        if len(enabled_trackers) > 8:
            raise DomainError(code="INVALID_ARGUMENT", message="enabled_trackers must have at most 8 items.", details={"max": 8, "given": len(enabled_trackers)})

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
        if fps not in {20, 30, 45, 60}:
            raise DomainError(code="INVALID_ARGUMENT", message="fps must be one of 20/30/45/60.", details={"fps": fps})
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
            raise DomainError(code="INVALID_ARGUMENT", message="data must be an object.")

        # Strict per-mode validation (v1)
        args: list[float]
        try:
            if gaze_mode == "CenterPitchYaw":
                args = [float(data["pitch"]), float(data["yaw"])]
            elif gaze_mode == "CenterPitchYawDist":
                # Official docs use `distance` (meters). For compatibility, accept `distance` or `distance_m`.
                dist = data["distance_m"] if "distance_m" in data else data["distance"]
                args = [float(data["pitch"]), float(data["yaw"]), float(dist)]
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
                message="data is missing required fields for the given gaze_mode.",
                details={"gaze_mode": gaze_mode, "missing": str(e)},
            ) from e
        except Exception as e:  # noqa: BLE001
            raise DomainError(code="INVALID_ARGUMENT", message="data fields must be numbers.", details={"gaze_mode": gaze_mode}) from e

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

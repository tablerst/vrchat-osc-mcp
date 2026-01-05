from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import AvatarSchema, ParameterSchema, ParameterType


def _as_param_type(v: Any) -> ParameterType:
    if v in ("Bool", "Int", "Float"):
        return v
    raise ValueError(f"unknown parameter type: {v!r}")


def load_avatar_schema(path: Path) -> AvatarSchema:
    """Load VRChat avatar OSC config JSON and build a lookup-friendly schema."""

    # VRChat avatar OSC config files (and the sample.json provided) may include a UTF-8 BOM.
    # json.loads rejects BOM when the input is a str, so we decode with utf-8-sig.
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("avatar config must be a JSON object")

    avatar_id = data.get("id")
    if not isinstance(avatar_id, str) or not avatar_id:
        raise ValueError("avatar config missing string field: id")

    avatar_name = data.get("name")
    if avatar_name is not None and not isinstance(avatar_name, str):
        avatar_name = None

    params_raw = data.get("parameters")
    if not isinstance(params_raw, list):
        raise ValueError("avatar config missing list field: parameters")

    parameters: dict[str, ParameterSchema] = {}
    by_suffix: dict[str, str] = {}

    for item in params_raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue

        input_obj = item.get("input")
        output_obj = item.get("output")

        input_address: str | None = None
        output_address: str | None = None
        p_type: ParameterType | None = None

        if isinstance(input_obj, dict):
            addr = input_obj.get("address")
            typ = input_obj.get("type")
            if isinstance(addr, str):
                input_address = addr
            if typ is not None:
                p_type = _as_param_type(typ)

        if isinstance(output_obj, dict):
            addr = output_obj.get("address")
            typ = output_obj.get("type")
            if isinstance(addr, str):
                output_address = addr
            if p_type is None and typ is not None:
                p_type = _as_param_type(typ)

        if p_type is None:
            # Skip unknown/untyped
            continue

        param = ParameterSchema(
            name=name,
            type=p_type,
            input_address=input_address,
            output_address=output_address,
        )
        parameters[name] = param

        for addr in (input_address, output_address):
            if not addr or not addr.startswith("/avatar/parameters/"):
                continue
            suffix = addr.removeprefix("/avatar/parameters/")
            # First win is fine; collisions are rare and not critical.
            by_suffix.setdefault(suffix, name)

    return AvatarSchema(
        avatar_id=avatar_id,
        avatar_name=avatar_name,
        source_path=path,
        parameters=parameters,
        _by_address_suffix=by_suffix,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ParameterType = Literal["Bool", "Int", "Float"]


@dataclass(frozen=True)
class ParameterSchema:
    name: str
    type: ParameterType
    input_address: str | None
    output_address: str | None

    @property
    def writable(self) -> bool:
        return self.input_address is not None


@dataclass(frozen=True)
class AvatarSchema:
    avatar_id: str
    avatar_name: str | None
    source_path: Path | None
    parameters: dict[str, ParameterSchema]
    _by_address_suffix: dict[str, str]

    def resolve(self, key: str) -> ParameterSchema | None:
        """Resolve by parameter name, or by OSC address suffix.

        VRChat config sometimes sanitizes the OSC address segment, so the OSC
        address suffix (after /avatar/parameters/) may not equal the "name" field.

        We accept:
        - exact parameter name
        - exact OSC address (input/output)
        - OSC suffix (e.g. "Foo_Bar")
        """

        if key in self.parameters:
            return self.parameters[key]

        # If user passed full address
        if key.startswith("/avatar/parameters/"):
            suffix = key.removeprefix("/avatar/parameters/")
            name = self._by_address_suffix.get(suffix)
            return self.parameters.get(name) if name else None

        # If user passed suffix
        name = self._by_address_suffix.get(key)
        return self.parameters.get(name) if name else None

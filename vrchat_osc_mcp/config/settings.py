from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.config import ConfigDict


class SSESettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class HTTPSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    path: str = "/mcp"


class MCPSettings(BaseModel):
    transport: Literal["stdio", "sse", "http"] = "stdio"
    sse: SSESettings = Field(default_factory=SSESettings)
    http: HTTPSettings = Field(default_factory=HTTPSettings)


class OSCEndpoint(BaseModel):
    ip: str = "127.0.0.1"
    port: int = 9000

    @field_validator("port")
    @classmethod
    def _port_range(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("port must be in [1, 65535]")
        return v


class OSCReceiveSettings(BaseModel):
    enabled: bool = False
    ip: str = "127.0.0.1"
    port: int = 9001

    @field_validator("port")
    @classmethod
    def _port_range(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("port must be in [1, 65535]")
        return v


class OSCSettings(BaseModel):
    send: OSCEndpoint = Field(default_factory=OSCEndpoint)
    receive: OSCReceiveSettings = Field(default_factory=OSCReceiveSettings)


class VRChatSettings(BaseModel):
    # Optional override for: LocalLow\\VRChat\\VRChat\\OSC
    osc_root: Path | None = None

    # Optional explicit avatar OSC config json path (e.g. avtr_*.json)
    avatar_config: Path | None = None


class LoggingSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = Field(True, alias="json")


class SafetySettings(BaseModel):
    osc_per_second: int = Field(60, ge=1, le=500)
    chat_per_minute: int = Field(10, ge=1, le=120)

    # Input safety valves
    max_axis_duration_ms: int = Field(2000, ge=50, le=5000)
    max_button_hold_ms: int = Field(1000, ge=20, le=5000)

    # Parameter policy (MVP-0 uses allowlist; MVP-1 upgrades to schema-backed strict)
    parameter_policy: Literal["strict", "allowlist", "permissive"] = "allowlist"
    allowed_parameters: list[str] = Field(default_factory=list)


class ProfilesSettings(BaseModel):
    directory: Path = Path("profiles")
    active: str = "default"


class Settings(BaseModel):
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    osc: OSCSettings = Field(default_factory=OSCSettings)
    vrchat: VRChatSettings = Field(default_factory=VRChatSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    profiles: ProfilesSettings = Field(default_factory=ProfilesSettings)

    @model_validator(mode="after")
    def _receiver_disabled_in_mvp0(self) -> "Settings":
        # MVP-0: receiver is optional but defaults to disabled.
        # If enabled, we keep config but server may still choose not to start it.
        return self

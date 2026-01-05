"""VRChat Official OSC MCP Server.

This project intentionally keeps VRChat-specific safety semantics in the domain
layer (auto-reset, clamping, trimming, rate limiting).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

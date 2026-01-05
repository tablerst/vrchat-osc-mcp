"""Repo entrypoint.

Keep this file tiny so `uv run python main.py` works, while the real
implementation lives in the `vrchat_osc_mcp` package.
"""

from vrchat_osc_mcp.main import main


if __name__ == "__main__":
    raise SystemExit(main())

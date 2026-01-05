from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from urllib.parse import urlparse

import httpx

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="list_tools_http",
        description=(
            "Connect to an MCP Streamable HTTP endpoint and list tools/prompts/resources with schemas. "
            "Designed for vrchat-osc-mcp, but works with any MCP server that supports Streamable HTTP."
        ),
    )
    p.add_argument(
        "--url",
        default="http://127.0.0.1:8001/mcp",
        help=(
            "MCP Streamable HTTP endpoint URL. "
            "If you pass a base like http://127.0.0.1:8001, the script will assume /mcp. "
            "(default: http://127.0.0.1:8001/mcp)"
        ),
    )
    p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP connect timeout seconds (best-effort). Default: 30.",
    )
    p.add_argument(
        "--sse-read-timeout",
        type=float,
        default=300.0,
        help="SSE read timeout seconds (best-effort). Default: 300.",
    )
    p.add_argument(
        "--show-capabilities",
        action="store_true",
        help="Also print initialize() capabilities/serverInfo.",
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write JSON files into (default: repo root).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write JSON files (stdout only).",
    )
    return p


def _normalize_url(url: str) -> str:
    """Allow passing a base URL like http://127.0.0.1:8001 (assume /mcp)."""

    p = urlparse(url)
    # urlparse("http://host:port") -> path == ""
    if p.path in ("", "/"):
        return url.rstrip("/") + "/mcp"
    return url


def _repo_root() -> Path:
    # scripts/list_tools_http.py -> repo root is parents[1]
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_payload_files(*, out_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}

    # Split files (requested)
    for key, filename in [
        ("tools", "mcp_tools.json"),
        ("prompts", "mcp_prompts.json"),
        ("resources", "mcp_resources.json"),
        ("resource_templates", "mcp_resource_templates.json"),
    ]:
        p = out_dir / filename
        _write_json(p, payload.get(key, []))
        written[key] = str(p)

    # Also write a single combined snapshot for convenience
    combined = out_dir / "mcp_catalog.json"
    _write_json(combined, payload)
    written["catalog"] = str(combined)

    return written


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of MCP/Pydantic-ish objects into JSON-serializable data."""

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}

    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:  # noqa: BLE001
            pass

    # Common attrs
    if hasattr(obj, "__dict__"):
        try:
            return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
        except Exception:  # noqa: BLE001
            pass

    return str(obj)


async def _paginate(
    fetch_page: Callable[[str | None], Any],
) -> list[Any]:
    items: list[Any] = []
    cursor: str | None = None

    while True:
        page = await fetch_page(cursor)
        # MCP types typically have .items + .nextCursor
        page_items = getattr(page, "items", None)
        if page_items is None:
            # Some endpoints use tool/resource specific fields.
            # Fallback to finding the first list-typed attribute.
            for k, v in vars(page).items():
                if isinstance(v, list):
                    page_items = v
                    break

        if page_items:
            items.extend(page_items)

        next_cursor = getattr(page, "nextCursor", None)
        if not next_cursor:
            break
        cursor = next_cursor

    return items


async def _collect(url: str, *, terminate_on_close: bool = True) -> dict[str, Any]:
    async with streamable_http_client(url, terminate_on_close=terminate_on_close) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()

            tools = await _paginate(lambda c: session.list_tools(cursor=c))
            prompts = await _paginate(lambda c: session.list_prompts(cursor=c))
            resources = await _paginate(lambda c: session.list_resources(cursor=c))
            resource_templates = await _paginate(lambda c: session.list_resource_templates(cursor=c))

            return {
                "initialize": init,
                "tools": tools,
                "prompts": prompts,
                "resources": resources,
                "resource_templates": resource_templates,
            }


def _print_text(data: dict[str, Any], *, show_capabilities: bool) -> None:
    init = data.get("initialize")

    if show_capabilities and init is not None:
        print("== initialize ==")
        init_j = _to_jsonable(init)
        # Keep it readable; still show full JSON.
        print(json.dumps(init_j, ensure_ascii=False, indent=2, sort_keys=True))
        print()

    tools = data.get("tools") or []
    prompts = data.get("prompts") or []
    resources = data.get("resources") or []
    resource_templates = data.get("resource_templates") or []

    print(f"== tools ({len(tools)}) ==")
    for t in tools:
        tj = _to_jsonable(t)
        name = tj.get("name")
        print(f"- {name}")
        if tj.get("description"):
            print(f"  desc: {tj['description']}")
        schema = tj.get("inputSchema") or tj.get("input_schema") or tj.get("parameters")
        if schema is not None:
            print("  inputSchema:")
            print("\n".join(["    " + line for line in json.dumps(schema, ensure_ascii=False, indent=2).splitlines()]))
        print()

    print(f"== prompts ({len(prompts)}) ==")
    for p in prompts:
        pj = _to_jsonable(p)
        name = pj.get("name")
        print(f"- {name}")
        if pj.get("description"):
            print(f"  desc: {pj['description']}")
        args = pj.get("arguments")
        if args is not None:
            print("  arguments:")
            print("\n".join(["    " + line for line in json.dumps(args, ensure_ascii=False, indent=2).splitlines()]))
        print()

    print(f"== resources ({len(resources)}) ==")
    for r in resources:
        rj = _to_jsonable(r)
        uri = rj.get("uri")
        print(f"- {uri}")
        if rj.get("name"):
            print(f"  name: {rj['name']}")
        if rj.get("description"):
            print(f"  desc: {rj['description']}")
        if rj.get("mimeType"):
            print(f"  mimeType: {rj['mimeType']}")
        print()

    print(f"== resource_templates ({len(resource_templates)}) ==")
    for rt in resource_templates:
        rtj = _to_jsonable(rt)
        tpl = rtj.get("uriTemplate") or rtj.get("uri_template")
        print(f"- {tpl}")
        if rtj.get("name"):
            print(f"  name: {rtj['name']}")
        if rtj.get("description"):
            print(f"  desc: {rtj['description']}")
        if rtj.get("mimeType"):
            print(f"  mimeType: {rtj['mimeType']}")
        print()


async def _main_async(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Note: timeout/sse-read-timeout are currently best-effort; the official client creates
    # a default httpx client with recommended timeouts unless you pass your own.
    # We keep flags to document intent and for future enhancement.
    _ = args.timeout
    _ = args.sse_read_timeout

    def _walk_exceptions(exc: BaseException) -> list[BaseException]:
        # Flatten ExceptionGroup trees so we can find the real root cause.
        if isinstance(exc, BaseExceptionGroup):
            out: list[BaseException] = []
            for sub in exc.exceptions:
                out.extend(_walk_exceptions(sub))
            return out
        return [exc]

    url = _normalize_url(args.url)

    try:
        data = await _collect(url)
    except BaseExceptionGroup as eg:
        flat = _walk_exceptions(eg)

        status_err = next((e for e in flat if isinstance(e, httpx.HTTPStatusError)), None)
        if status_err is not None:
            msg = f"HTTP {status_err.response.status_code} when calling {status_err.request.method} {status_err.request.url}"
            print("ERROR:", msg)
            print(
                "Hint: make sure the MCP server is running with --transport http and the URL path matches (--http-path). "
                "If the server failed to bind the port, you may be hitting a different service on that URL."
            )
            return 2

        http_err = next((e for e in flat if isinstance(e, httpx.HTTPError)), None)
        if http_err is not None:
            print("ERROR: HTTP client error:", str(http_err))
            print("Hint: check that the URL is reachable, and that no proxy is intercepting localhost.")
            return 2

        # Fallback: show the first exception in the group.
        first = flat[0] if flat else eg
        print("ERROR:", str(first))
        return 2
    except httpx.HTTPStatusError as e:
        msg = f"HTTP {e.response.status_code} when calling {e.request.method} {e.request.url}"
        print("ERROR:", msg)
        print("Hint: make sure the MCP server is running with --transport http and the URL path matches (--http-path).")
        return 2
    except httpx.HTTPError as e:
        print("ERROR: HTTP client error:", str(e))
        print("Hint: check that the URL is reachable, and that no proxy is intercepting localhost.")
        return 2
    except Exception as e:  # noqa: BLE001
        print("ERROR:", str(e))
        return 2

    if args.format == "json":
        payload = {
            "initialize": _to_jsonable(data.get("initialize")),
            "tools": [_to_jsonable(x) for x in (data.get("tools") or [])],
            "prompts": [_to_jsonable(x) for x in (data.get("prompts") or [])],
            "resources": [_to_jsonable(x) for x in (data.get("resources") or [])],
            "resource_templates": [_to_jsonable(x) for x in (data.get("resource_templates") or [])],
        }

        if not args.no_write:
            out_dir = args.output_dir or _repo_root()
            written = _save_payload_files(out_dir=out_dir, payload=payload)
            # Always emit a short summary so callers know where the files are.
            print(json.dumps({"url": url, "written": written}, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

        return 0

    _print_text(data, show_capabilities=bool(args.show_capabilities))
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())

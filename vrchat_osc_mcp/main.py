from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from .config.loader import load_settings
from .mcp_server import create_server
from .observability.logging import configure_logging, get_logger
from .osc.transport import OSCTransport
from .domain.adapter import VRChatDomainAdapter
from .vrc_config.parser import load_avatar_schema
from .vrc_config.resolver import resolve_avatar_config_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vrchat-osc-mcp")

    p.add_argument("--config", type=Path, default=None, help="YAML 配置文件路径（默认探测 ./config.yaml 或 ./config/config.yaml）")
    p.add_argument("--transport", choices=["stdio", "sse", "http"], default=None)

    p.add_argument("--osc-send-ip", default=None)
    p.add_argument("--osc-send-port", type=int, default=None)

    p.add_argument("--enable-receiver", action="store_true", help="启用 OSC receiver（MVP-0 默认不启动）")
    p.add_argument("--no-receiver", action="store_true", help="禁用 OSC receiver")
    p.add_argument("--osc-receive-ip", default=None)
    p.add_argument("--osc-receive-port", type=int, default=None)

    p.add_argument("--sse-host", default=None)
    p.add_argument("--sse-port", type=int, default=None)

    p.add_argument("--http-host", default=None)
    p.add_argument("--http-port", type=int, default=None)
    p.add_argument("--http-path", default=None)

    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)

    p.add_argument("--vrchat-osc-root", type=Path, default=None)
    p.add_argument("--avatar-config", type=Path, default=None, help="显式指定 Avatar OSC config JSON（avtr_*.json）；用于 schema 严格校验")

    return p


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    o: dict[str, Any] = {}

    if args.transport is not None:
        o.setdefault("mcp", {})["transport"] = args.transport

    if args.sse_host is not None or args.sse_port is not None:
        o.setdefault("mcp", {}).setdefault("sse", {})
        if args.sse_host is not None:
            o["mcp"]["sse"]["host"] = args.sse_host
        if args.sse_port is not None:
            o["mcp"]["sse"]["port"] = args.sse_port

    if args.http_host is not None or args.http_port is not None or args.http_path is not None:
        o.setdefault("mcp", {}).setdefault("http", {})
        if args.http_host is not None:
            o["mcp"]["http"]["host"] = args.http_host
        if args.http_port is not None:
            o["mcp"]["http"]["port"] = args.http_port
        if args.http_path is not None:
            o["mcp"]["http"]["path"] = args.http_path

    if args.osc_send_ip is not None or args.osc_send_port is not None:
        o.setdefault("osc", {}).setdefault("send", {})
        if args.osc_send_ip is not None:
            o["osc"]["send"]["ip"] = args.osc_send_ip
        if args.osc_send_port is not None:
            o["osc"]["send"]["port"] = args.osc_send_port

    # Receiver tri-state: CLI overrides YAML when explicitly specified.
    if args.enable_receiver and args.no_receiver:
        raise SystemExit("--enable-receiver 与 --no-receiver 不能同时使用")

    if args.enable_receiver or args.no_receiver or args.osc_receive_ip is not None or args.osc_receive_port is not None:
        o.setdefault("osc", {}).setdefault("receive", {})
        if args.enable_receiver:
            o["osc"]["receive"]["enabled"] = True
        if args.no_receiver:
            o["osc"]["receive"]["enabled"] = False
        if args.osc_receive_ip is not None:
            o["osc"]["receive"]["ip"] = args.osc_receive_ip
        if args.osc_receive_port is not None:
            o["osc"]["receive"]["port"] = args.osc_receive_port

    if args.log_level is not None:
        o.setdefault("logging", {})["level"] = args.log_level

    if args.vrchat_osc_root is not None:
        o.setdefault("vrchat", {})["osc_root"] = str(args.vrchat_osc_root)

    if args.avatar_config is not None:
        o.setdefault("vrchat", {})["avatar_config"] = str(args.avatar_config)

    return o


async def _run(settings) -> None:
    configure_logging(level=settings.logging.level, json_logs=settings.logging.json_logs)
    logger = get_logger().bind(component="app")

    schema = None
    schema_path = resolve_avatar_config_path(
        osc_root=settings.vrchat.osc_root,
        explicit_path=settings.vrchat.avatar_config,
    )
    if schema_path is not None:
        try:
            schema = load_avatar_schema(schema_path)
            logger.info(
                "schema.loaded",
                schema_source="local_config",
                schema_path=str(schema_path),
                avatar_id=schema.avatar_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "schema.load_failed",
                schema_source="local_config",
                schema_path=str(schema_path),
                error=str(e),
            )

    osc = OSCTransport(
        send_ip=settings.osc.send.ip,
        send_port=settings.osc.send.port,
        osc_per_second=settings.safety.osc_per_second,
        logger=get_logger().bind(component="osc"),
    )
    await osc.start()

    adapter = VRChatDomainAdapter(
        transport=osc,
        settings=settings,
        logger=get_logger().bind(component="domain"),
        schema=schema,
    )

    mcp = create_server(adapter=adapter)

    logger.info(
        "server.start",
        transport=settings.mcp.transport,
        osc_send_ip=settings.osc.send.ip,
        osc_send_port=settings.osc.send.port,
        sse_host=settings.mcp.sse.host,
        sse_port=settings.mcp.sse.port,
        http_host=settings.mcp.http.host,
        http_port=settings.mcp.http.port,
        http_path=settings.mcp.http.path,
    )

    try:
        if settings.mcp.transport == "stdio":
            await mcp.run_async(transport="stdio")
        elif settings.mcp.transport == "sse":
            await mcp.run_async(transport="sse", host=settings.mcp.sse.host, port=settings.mcp.sse.port)
        else:
            await mcp.run_async(
                transport="http",
                host=settings.mcp.http.host,
                port=settings.mcp.http.port,
                path=settings.mcp.http.path,
            )
    finally:
        await osc.close()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    loaded = load_settings(project_root=project_root, config_path=args.config, cli_overrides=_cli_overrides(args))

    asyncio.run(_run(loaded.settings))
    return 0

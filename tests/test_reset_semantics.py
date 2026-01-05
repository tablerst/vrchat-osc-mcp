from __future__ import annotations

import asyncio
import socket

import pytest
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

from vrchat_osc_mcp.config.settings import Settings
from vrchat_osc_mcp.domain.adapter import VRChatDomainAdapter
from vrchat_osc_mcp.observability.logging import configure_logging, get_logger
from vrchat_osc_mcp.osc.transport import OSCTransport


class OscCapture:
    def __init__(self) -> None:
        self.messages: asyncio.Queue[tuple[str, tuple]] = asyncio.Queue()

    def handler(self, address: str, *args):
        self.messages.put_nowait((address, args))


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.asyncio
async def test_input_tap_sends_1_then_0():
    configure_logging(level="ERROR", json_logs=True)

    capture = OscCapture()
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(capture.handler)

    port = _free_udp_port()
    server = AsyncIOOSCUDPServer(("127.0.0.1", port), dispatcher, asyncio.get_running_loop())
    transport, _protocol = await server.create_serve_endpoint()

    osc = None
    try:
        settings = Settings.model_validate(
            {
                "osc": {"send": {"ip": "127.0.0.1", "port": port}},
                "safety": {"osc_per_second": 500},
            }
        )

        osc = OSCTransport(
            send_ip=settings.osc.send.ip,
            send_port=settings.osc.send.port,
            osc_per_second=settings.safety.osc_per_second,
            logger=get_logger().bind(component="test-osc"),
        )
        await osc.start()

        adapter = VRChatDomainAdapter(transport=osc, settings=settings, logger=get_logger().bind(component="test-domain"))
        await adapter.input_tap(button="Jump", hold_ms=50, trace_id="t")

        a1, args1 = await asyncio.wait_for(capture.messages.get(), timeout=1)
        a2, args2 = await asyncio.wait_for(capture.messages.get(), timeout=1)

        assert a1 == "/input/Jump"
        assert args1 == (1,)
        assert a2 == "/input/Jump"
        assert args2 == (0,)

    finally:
        if osc is not None:
            await osc.close()
        transport.close()


@pytest.mark.asyncio
async def test_input_axis_clamps_and_resets():
    configure_logging(level="ERROR", json_logs=True)

    capture = OscCapture()
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(capture.handler)

    port = _free_udp_port()
    server = AsyncIOOSCUDPServer(("127.0.0.1", port), dispatcher, asyncio.get_running_loop())
    transport, _protocol = await server.create_serve_endpoint()

    osc = None
    try:
        settings = Settings.model_validate(
            {
                "osc": {"send": {"ip": "127.0.0.1", "port": port}},
                "safety": {"osc_per_second": 500, "max_axis_duration_ms": 2000},
            }
        )

        osc = OSCTransport(
            send_ip=settings.osc.send.ip,
            send_port=settings.osc.send.port,
            osc_per_second=settings.safety.osc_per_second,
            logger=get_logger().bind(component="test-osc"),
        )
        await osc.start()

        adapter = VRChatDomainAdapter(transport=osc, settings=settings, logger=get_logger().bind(component="test-domain"))
        await adapter.input_axis(axis="Vertical", value=2.0, duration_ms=30, trace_id="t")

        a1, args1 = await asyncio.wait_for(capture.messages.get(), timeout=1)
        a2, args2 = await asyncio.wait_for(capture.messages.get(), timeout=1)

        assert a1 == "/input/Vertical"
        assert args1 == (1.0,)
        assert a2 == "/input/Vertical"
        assert args2 == (0.0,)

    finally:
        if osc is not None:
            await osc.close()
        transport.close()

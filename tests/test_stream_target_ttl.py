from __future__ import annotations

import asyncio
import socket
from typing import cast

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


async def _wait_for_message(
    capture: OscCapture,
    *,
    predicate,
    timeout_s: float = 1.5,
) -> tuple[str, tuple]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last: tuple[str, tuple] | None = None
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise AssertionError(f"timeout waiting for message; last={last!r}")
        last = await asyncio.wait_for(capture.messages.get(), timeout=remaining)
        if predicate(last):
            return last


@pytest.mark.asyncio
async def test_tracking_stream_stale_target_is_neutralized():
    configure_logging(level="ERROR", json_logs=True)

    capture = OscCapture()
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(capture.handler)

    port = _free_udp_port()
    loop = cast(asyncio.BaseEventLoop, asyncio.get_running_loop())
    server = AsyncIOOSCUDPServer(("127.0.0.1", port), dispatcher, loop)
    transport, _protocol = await server.create_serve_endpoint()

    osc = None
    try:
        settings = Settings.model_validate(
            {
                "osc": {"send": {"ip": "127.0.0.1", "port": port}},
                "safety": {
                    "osc_per_second": 500,
                    "tracking_target_ttl_ms": 100,
                },
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

        await adapter.tracking_stream_start(
            fps=60,
            enabled_trackers=["hip"],
            neutral_on_stop=False,
            trace_id="t",
        )
        await adapter.tracking_set_tracker_pose(
            tracker="hip",
            position_m={"x": 1.0, "y": 2.0, "z": 3.0},
            rotation_euler_deg={"x": 10.0, "y": 20.0, "z": 30.0},
            trace_id="t",
        )

        # See at least one non-neutral frame while target is fresh.
        await _wait_for_message(
            capture,
            predicate=lambda m: m[0] == "/tracking/trackers/1/position" and m[1] == (1.0, 2.0, 3.0),
        )
        await _wait_for_message(
            capture,
            predicate=lambda m: m[0] == "/tracking/trackers/1/rotation" and m[1] == (10.0, 20.0, 30.0),
        )

        # After TTL, the stream should emit a single neutral frame and clear the target.
        await asyncio.sleep(0.25)
        await _wait_for_message(
            capture,
            predicate=lambda m: m[0] == "/tracking/trackers/1/position" and m[1] == (0.0, 0.0, 0.0),
        )
        await _wait_for_message(
            capture,
            predicate=lambda m: m[0] == "/tracking/trackers/1/rotation" and m[1] == (0.0, 0.0, 0.0),
        )

    finally:
        try:
            if osc is not None:
                await osc.close()
        finally:
            transport.close()


@pytest.mark.asyncio
async def test_eye_stream_stale_targets_revert_to_neutral():
    configure_logging(level="ERROR", json_logs=True)

    capture = OscCapture()
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(capture.handler)

    port = _free_udp_port()
    loop = cast(asyncio.BaseEventLoop, asyncio.get_running_loop())
    server = AsyncIOOSCUDPServer(("127.0.0.1", port), dispatcher, loop)
    transport, _protocol = await server.create_serve_endpoint()

    osc = None
    try:
        settings = Settings.model_validate(
            {
                "osc": {"send": {"ip": "127.0.0.1", "port": port}},
                "safety": {
                    "osc_per_second": 500,
                    "eye_target_ttl_ms": 100,
                },
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

        await adapter.eye_stream_start(
            fps=60,
            gaze_mode="CenterPitchYaw",
            neutral_on_stop=False,
            trace_id="t",
        )

        await adapter.eye_set_blink(amount=1.0, trace_id="t")
        await adapter.eye_set_gaze(gaze_mode="CenterPitchYaw", data={"pitch": 0.25, "yaw": -0.5}, trace_id="t")

        # See non-neutral keepalive.
        await _wait_for_message(
            capture,
            predicate=lambda m: m[0] == "/tracking/eye/EyesClosedAmount" and m[1] == (1.0,),
        )
        await _wait_for_message(
            capture,
            predicate=lambda m: m[0] == "/tracking/eye/CenterPitchYaw" and m[1] == (0.25, -0.5),
        )

        # After TTL, the stream should revert to neutral values.
        await asyncio.sleep(0.25)
        await _wait_for_message(
            capture,
            predicate=lambda m: m[0] == "/tracking/eye/EyesClosedAmount" and m[1] == (0.0,),
        )
        await _wait_for_message(
            capture,
            predicate=lambda m: m[0] == "/tracking/eye/CenterPitchYaw" and m[1] == (0.0, 0.0),
        )

    finally:
        try:
            if osc is not None:
                await osc.close()
        finally:
            transport.close()

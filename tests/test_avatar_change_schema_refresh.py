from __future__ import annotations

import asyncio
import socket

import pytest

from vrchat_osc_mcp.config.settings import Settings
from vrchat_osc_mcp.domain.adapter import VRChatDomainAdapter
from vrchat_osc_mcp.observability.logging import configure_logging, get_logger
from vrchat_osc_mcp.osc.transport import OSCTransport


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.asyncio
async def test_avatar_change_refreshes_schema_from_local_config(tmp_path):
    configure_logging(level="ERROR", json_logs=True)

    osc_root = tmp_path / "OSC"
    cfg = osc_root / "usr_123" / "Avatars" / "avtr_test.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        '{"id":"avtr_test","name":"Test","parameters":[{"name":"Foo","input":{"address":"/avatar/parameters/Foo","type":"Bool"}}]}',
        encoding="utf-8",
    )

    port = _free_udp_port()
    settings = Settings.model_validate(
        {
            "osc": {"send": {"ip": "127.0.0.1", "port": port}},
            "vrchat": {"osc_root": str(osc_root)},
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

    try:
        adapter = VRChatDomainAdapter(transport=osc, settings=settings, logger=get_logger().bind(component="test-domain"))
        await adapter.on_avatar_change("avtr_test")

        caps = adapter.meta_get_capabilities(refresh=False)
        avatar = caps["avatar"]
        assert avatar["current_avatar_id"] == "avtr_test"
        assert avatar["schema_avatar_id"] == "avtr_test"
        assert avatar["schema_stale"] is False
        assert avatar["schema_last_error"] is None
        assert avatar["schema_source"] == "local_config_by_avatar_id"

        params = adapter.avatar_list_parameters()["parameters"]
        assert any(p["name"] == "Foo" and p["type"] == "Bool" for p in params)

    finally:
        await osc.close()


@pytest.mark.asyncio
async def test_avatar_change_refresh_failure_keeps_last_good_schema(tmp_path):
    configure_logging(level="ERROR", json_logs=True)

    osc_root = tmp_path / "OSC"
    cfg = osc_root / "usr_123" / "Avatars" / "avtr_a.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        '{"id":"avtr_a","name":"A","parameters":[{"name":"Foo","input":{"address":"/avatar/parameters/Foo","type":"Bool"}}]}',
        encoding="utf-8",
    )

    port = _free_udp_port()
    settings = Settings.model_validate(
        {
            "osc": {"send": {"ip": "127.0.0.1", "port": port}},
            "vrchat": {"osc_root": str(osc_root)},
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

    try:
        adapter = VRChatDomainAdapter(transport=osc, settings=settings, logger=get_logger().bind(component="test-domain"))
        await adapter.on_avatar_change("avtr_a")
        await adapter.on_avatar_change("avtr_missing")

        caps = adapter.meta_get_capabilities(refresh=False)
        avatar = caps["avatar"]
        assert avatar["current_avatar_id"] == "avtr_missing"
        assert avatar["schema_avatar_id"] == "avtr_a"  # kept last-good
        assert avatar["schema_stale"] is True
        assert avatar["schema_last_error"] is not None

        params = adapter.avatar_list_parameters()["parameters"]
        assert any(p["name"] == "Foo" for p in params)

    finally:
        await osc.close()

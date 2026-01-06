from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer


AvatarChangeCallback = Callable[[str], Awaitable[None] | None]
ParameterCallback = Callable[[str, Any], Awaitable[None] | None]


class OSCReceiver:
    """Lightweight OSC UDP receiver.

    MVP-1 responsibilities:
    - receive /avatar/change and surface avatar_id to a callback
    - (optional) observe /avatar/parameters/* updates

    Notes:
    - VRChat default out port is 9001 (VRChat -> apps), so the receiver usually
      binds to 9001, but that port may be occupied by other tools.
    """

    def __init__(
        self,
        *,
        bind_ip: str,
        port: int,
        logger,
        on_avatar_change: AvatarChangeCallback | None = None,
        on_parameter: ParameterCallback | None = None,
    ) -> None:
        self._bind_ip = bind_ip
        self._port = int(port)
        self._logger = logger
        self._on_avatar_change = on_avatar_change
        self._on_parameter = on_parameter

        self._dispatcher = Dispatcher()
        self._server: AsyncIOOSCUDPServer | None = None
        self._transport = None
        self._protocol = None

        self._dispatcher.map("/avatar/change", self._handle_avatar_change)
        self._dispatcher.set_default_handler(self._handle_default)

    @property
    def bind_ip(self) -> str:
        return self._bind_ip

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        if self._transport is not None:
            return

        loop = asyncio.get_running_loop()
        # python-osc's type stubs expect BaseEventLoop; get_running_loop returns AbstractEventLoop.
        self._server = AsyncIOOSCUDPServer((self._bind_ip, self._port), self._dispatcher, loop)  # type: ignore[arg-type]
        self._transport, self._protocol = await self._server.create_serve_endpoint()

        self._logger.info(
            "osc.receiver.start",
            bind_ip=self._bind_ip,
            bind_port=self._port,
        )

    async def close(self) -> None:
        if self._transport is None:
            return

        try:
            self._logger.info(
                "osc.receiver.stop",
                bind_ip=self._bind_ip,
                bind_port=self._port,
            )
        finally:
            self._transport.close()
            self._transport = None
            self._protocol = None
            self._server = None

    # -----------------
    # OSC handlers
    # -----------------

    def _handle_avatar_change(self, address: str, *args: Any) -> None:
        avatar_id: str | None = None
        if args:
            v = args[0]
            if isinstance(v, bytes):
                try:
                    avatar_id = v.decode("utf-8", errors="replace")
                except Exception:
                    avatar_id = str(v)
            else:
                avatar_id = str(v)

        avatar_id = (avatar_id or "").strip() or None

        self._logger.info(
            "osc.recv.avatar_change",
            osc_address=address,
            avatar_id=avatar_id,
        )

        if avatar_id and self._on_avatar_change is not None:
            self._call_cb(self._on_avatar_change, avatar_id)

    def _handle_default(self, address: str, *args: Any) -> None:
        # Minimal support for parameter observation; mainly for future debugging.
        if address.startswith("/avatar/parameters/"):
            name = address.removeprefix("/avatar/parameters/")
            value: Any = args[0] if args else None
            if self._on_parameter is not None:
                self._call_cb(self._on_parameter, name, value)
            return

    @staticmethod
    def _call_cb(cb: Callable[..., Awaitable[None] | None], *args: Any) -> None:
        try:
            r = cb(*args)
            if asyncio.iscoroutine(r):
                asyncio.create_task(r)  # fire-and-forget
        except Exception:
            # Avoid crashing the OSC server on callback errors.
            # The owning application should log errors in the callback itself.
            return

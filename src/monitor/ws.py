"""WebSocket connection manager for monitor event fan-out."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class WSManager:
    """Track websocket clients and broadcast events."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, socket: WebSocket) -> None:
        await socket.accept()
        async with self._lock:
            self._clients.add(socket)

    async def disconnect(self, socket: WebSocket) -> None:
        async with self._lock:
            if socket in self._clients:
                self._clients.remove(socket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)
        for socket in clients:
            try:
                await socket.send_json(payload)
            except Exception:
                await self.disconnect(socket)

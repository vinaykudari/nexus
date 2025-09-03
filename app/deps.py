from __future__ import annotations

import asyncio
from typing import AsyncIterator, Literal

import httpx
from fastapi import Depends, Header, HTTPException, status

from .config import settings


_client_lock = asyncio.Lock()
_async_client: httpx.AsyncClient | None = None


async def get_async_client() -> AsyncIterator[httpx.AsyncClient]:
    global _async_client
    if _async_client is None:
        async with _client_lock:
            if _async_client is None:
                _async_client = httpx.AsyncClient(
                    timeout=settings.http_timeout_seconds,
                    limits=httpx.Limits(
                        max_connections=settings.http_max_connections,
                        max_keepalive_connections=settings.http_max_keepalive_connections,
                    ),
                )
    yield _async_client


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    return x_api_key


ProviderName = Literal["openai", "anthropic"]


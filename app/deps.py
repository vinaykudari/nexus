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
    provider: "ProviderName",  # pulled from path param
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    # Prefer header if present; otherwise fall back to env-configured key per provider
    if x_api_key:
        return x_api_key
    # Fallbacks from environment via settings
    if provider == "openai" and settings.openai_api_key:
        return settings.openai_api_key
    if provider == "anthropic" and settings.anthropic_api_key:
        return settings.anthropic_api_key
    if provider == "gemini" and settings.gemini_api_key:
        return settings.gemini_api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing API key: provide 'X-API-Key' header or set PROVIDER env (OPENAI_API_KEY | ANTHROPIC_API_KEY | GEMINI_API_KEY)",
    )


ProviderName = Literal["openai", "anthropic", "gemini"]

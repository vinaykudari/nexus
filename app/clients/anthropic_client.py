from __future__ import annotations

from typing import Any, AsyncIterator, Dict

import httpx
from fastapi import HTTPException, status

from ..config import settings
from ..models import ChatRequest
from .base import ChatProvider


def _to_anthropic_messages(req: ChatRequest) -> Dict[str, Any]:
    system = None
    messages = []
    for m in req.messages:
        if m.role == "system":
            # Anthropic uses top-level system, keep last one
            system = m.content
        elif m.role in ("user", "assistant"):
            messages.append({"role": m.role, "content": m.content})
    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    return payload


class AnthropicProvider(ChatProvider):
    name = "anthropic"

    async def chat(self, api_key: str, req: ChatRequest) -> Dict[str, Any]:
        if req.max_tokens is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Anthropic requires 'max_tokens' to be set",
            )
        url = f"{settings.anthropic_base_url}/messages"
        payload = _to_anthropic_messages(req)
        headers = {
            "x-api-key": api_key,
            "anthropic-version": settings.anthropic_version,
        }
        resp = await self.client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Normalize minimal response
        model = data.get("model", req.model)
        content = ""
        blocks = data.get("content") or []
        if isinstance(blocks, list) and blocks:
            # Newer API returns list of blocks with type "text"
            pieces = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "text":
                    t = b.get("text")
                    if isinstance(t, str):
                        pieces.append(t)
            content = "".join(pieces)
        stop_reason = data.get("stop_reason")
        return {
            "provider": self.name,
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
        }

    async def stream(self, api_key: str, req: ChatRequest) -> AsyncIterator[bytes]:
        if req.max_tokens is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Anthropic requires 'max_tokens' to be set",
            )
        url = f"{settings.anthropic_base_url}/messages"
        payload = _to_anthropic_messages(req)
        payload["stream"] = True
        headers = {
            "x-api-key": api_key,
            "anthropic-version": settings.anthropic_version,
        }

        async with self.client.stream("POST", url, json=payload, headers=headers) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                # Anthropic uses SSE with data: lines as well
                yield chunk

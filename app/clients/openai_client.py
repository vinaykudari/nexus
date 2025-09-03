from __future__ import annotations

from typing import Any, AsyncIterator, Dict

import httpx

from ..config import settings
from ..models import ChatRequest
from .base import ChatProvider


class OpenAIProvider(ChatProvider):
    name = "openai"

    async def chat(self, api_key: str, req: ChatRequest) -> Dict[str, Any]:
        url = f"{settings.openai_base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
        }
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature

        headers = {"Authorization": f"Bearer {api_key}"}
        resp = await self.client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Normalize minimal response
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        stop_reason = data.get("choices", [{}])[0].get("finish_reason")
        return {
            "provider": self.name,
            "model": data.get("model", req.model),
            "content": content,
            "stop_reason": stop_reason,
        }

    async def stream(self, api_key: str, req: ChatRequest) -> AsyncIterator[bytes]:
        url = f"{settings.openai_base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
            "stream": True,
        }
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature

        headers = {"Authorization": f"Bearer {api_key}"}

        async with self.client.stream("POST", url, json=payload, headers=headers) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                # Pass-through chunks (SSE formatted by OpenAI)
                yield chunk


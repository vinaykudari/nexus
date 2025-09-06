from __future__ import annotations

import base64
from typing import Any, AsyncIterator, Dict, List

import httpx

from ..config import settings
from ..models import ChatRequest, ContentPart, TextPart, ImageURLPart, ImageBase64Part
from .base import ChatProvider


def _system_from_messages(req: ChatRequest) -> List[Dict[str, Any]]:
    # Gemini "systemInstruction" supports parts; we only pass text parts for simplicity
    for m in req.messages:
        if m.role == "system":
            content = m.content
            if isinstance(content, str):
                return [{"text": content}]
            else:
                texts = [p.text for p in content if isinstance(p, TextPart)]
                if texts:
                    return [{"text": "".join(texts)}]
    return []


async def _to_gemini_parts(client: httpx.AsyncClient, parts: List[ContentPart]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in parts:
        if isinstance(p, TextPart):
            out.append({"text": p.text})
        elif isinstance(p, ImageBase64Part):
            out.append({"inlineData": {"mimeType": p.mime_type, "data": p.data}})
        elif isinstance(p, ImageURLPart):
            # Fetch and inline as base64 to avoid requiring Google Files API
            resp = await client.get(p.url)
            resp.raise_for_status()
            data_b64 = base64.b64encode(resp.content).decode("ascii")
            mime = p.mime_type or resp.headers.get("content-type", "image/png").split(";")[0]
            out.append({"inlineData": {"mimeType": mime, "data": data_b64}})
    return out


class GeminiProvider(ChatProvider):
    name = "gemini"

    async def chat(self, api_key: str, req: ChatRequest) -> Dict[str, Any]:
        url = f"{settings.gemini_base_url}/models/{req.model}:generateContent"
        contents: List[Dict[str, Any]] = []
        for m in req.messages:
            if m.role == "system":
                continue
            role = "user" if m.role == "user" else "model"
            parts = m.content if isinstance(m.content, list) else [TextPart(type="text", text=m.content)]
            gparts = await _to_gemini_parts(self.client, parts)
            contents.append({"role": role, "parts": gparts})

        payload: Dict[str, Any] = {"contents": contents}
        sys_parts = _system_from_messages(req)
        if sys_parts:
            payload["systemInstruction"] = {"parts": sys_parts}

        if req.max_tokens is not None:
            payload.setdefault("generationConfig", {})["maxOutputTokens"] = req.max_tokens
        if req.temperature is not None:
            payload.setdefault("generationConfig", {})["temperature"] = req.temperature

        params = {"key": api_key}
        resp = await self.client.post(url, json=payload, params=params)
        resp.raise_for_status()
        data = resp.json()

        # Normalize
        parts_out: List[ContentPart] = []
        candidates = data.get("candidates") or []
        if candidates:
            content = candidates[0].get("content") or {}
            for p in (content.get("parts") or []):
                if "text" in p and isinstance(p["text"], str):
                    parts_out.append(TextPart(type="text", text=p["text"]))
                elif "inlineData" in p and isinstance(p["inlineData"], dict):
                    mime = p["inlineData"].get("mimeType") or "image/png"
                    data_b64 = p["inlineData"].get("data")
                    if isinstance(data_b64, str):
                        parts_out.append(ImageBase64Part(type="image_base64", data=data_b64, mime_type=mime))

        text_joined = "".join([pt.text for pt in parts_out if isinstance(pt, TextPart)])

        return {
            "provider": self.name,
            "model": req.model,
            "content": text_joined,
            "content_parts": [p.model_dump() for p in parts_out],
            "stop_reason": data.get("finishReason") or None,
        }

    async def stream(self, api_key: str, req: ChatRequest) -> AsyncIterator[bytes]:
        url = f"{settings.gemini_base_url}/models/{req.model}:streamGenerateContent"
        contents: List[Dict[str, Any]] = []
        for m in req.messages:
            if m.role == "system":
                continue
            role = "user" if m.role == "user" else "model"
            parts = m.content if isinstance(m.content, list) else [TextPart(type="text", text=m.content)]
            gparts = await _to_gemini_parts(self.client, parts)
            contents.append({"role": role, "parts": gparts})

        payload: Dict[str, Any] = {"contents": contents}
        sys_parts = _system_from_messages(req)
        if sys_parts:
            payload["systemInstruction"] = {"parts": sys_parts}
        if req.max_tokens is not None:
            payload.setdefault("generationConfig", {})["maxOutputTokens"] = req.max_tokens
        if req.temperature is not None:
            payload.setdefault("generationConfig", {})["temperature"] = req.temperature

        params = {"key": api_key}
        async with self.client.stream("POST", url, json=payload, params=params) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                yield chunk


from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Path, status
from fastapi.responses import JSONResponse, StreamingResponse

from .deps import get_async_client, require_api_key, ProviderName
from .models import ChatRequest
from .clients.openai_client import OpenAIProvider
from .clients.anthropic_client import AnthropicProvider
from .clients.gemini_client import GeminiProvider

router = APIRouter(prefix="/v1", tags=["chat"])


def _provider_factory(provider: ProviderName, client) -> OpenAIProvider | AnthropicProvider | GeminiProvider:
    if provider == "openai":
        return OpenAIProvider(client)
    if provider == "anthropic":
        return AnthropicProvider(client)
    if provider == "gemini":
        return GeminiProvider(client)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported provider: {provider}",
    )


@router.post("/chat/{provider}")
async def chat(
    provider: ProviderName = Path(..., description="LLM provider: openai | anthropic | gemini"),
    body: ChatRequest = ...,  # Pydantic validation
    api_key: str = Depends(require_api_key),
    client=Depends(get_async_client),
):
    p = _provider_factory(provider, client)
    if body.stream:
        async def iterator() -> AsyncIterator[bytes]:
            async for chunk in p.stream(api_key, body):
                yield chunk

        return StreamingResponse(iterator(), media_type="text/event-stream")
    else:
        data = await p.chat(api_key, body)
        return JSONResponse(content=data)

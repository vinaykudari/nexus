from __future__ import annotations

from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str = Field(..., description="Upstream model name")
    messages: List[ChatMessage]
    stream: bool = Field(False, description="Enable streaming responses")
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, ge=0, le=2)


class ChatResponse(BaseModel):
    # Minimal normalized response for non-streaming
    provider: Literal["openai", "anthropic"]
    model: str
    content: str
    stop_reason: Optional[str] = None


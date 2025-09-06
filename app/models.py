from __future__ import annotations

from typing import Literal, List, Optional, Union, Annotated
from pydantic import BaseModel, Field


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageURLPart(BaseModel):
    type: Literal["image_url"]
    url: str
    mime_type: Optional[str] = None


class ImageBase64Part(BaseModel):
    type: Literal["image_base64"]
    data: str
    mime_type: str


ContentPart = Annotated[Union[TextPart, ImageURLPart, ImageBase64Part], Field(discriminator="type")]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    # Accept plain string for backwards compatibility, or a list of parts
    content: Union[str, List[ContentPart]]


class ChatRequest(BaseModel):
    model: str = Field(..., description="Upstream model name")
    messages: List[ChatMessage]
    stream: bool = Field(False, description="Enable streaming responses")
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, ge=0, le=2)


class ChatResponse(BaseModel):
    # Minimal normalized response for non-streaming
    provider: Literal["openai", "anthropic", "gemini"]
    model: str
    content: str
    content_parts: List[ContentPart] = Field(default_factory=list)
    stop_reason: Optional[str] = None

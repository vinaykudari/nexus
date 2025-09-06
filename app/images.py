from __future__ import annotations

import base64
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Header, status
from fastapi.responses import Response

from .deps import get_async_client
from .models import ChatMessage, ChatRequest, ImageBase64Part, TextPart
from .clients.gemini_client import GeminiProvider
from .clients.veo_client import VeoClient
from .config import settings


router = APIRouter(prefix="/v1/images", tags=["images"])

_rules_path = Path(__file__).with_name("rules.md")
_rules_text = _rules_path.read_text(encoding="utf-8") if _rules_path.exists() else ""
_video_rules_path = Path(__file__).with_name("rules_video.md")
_video_rules_text = _video_rules_path.read_text(encoding="utf-8") if _video_rules_path.exists() else ""


async def _require_google_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    if x_api_key:
        return x_api_key
    if settings.gemini_api_key:
        return settings.gemini_api_key
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key or GEMINI_API_KEY")


@router.post("/apply")
async def apply(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    model: str = Form("gemini-2.5-flash-image-preview"),
    video_model: str = Form("veo-3.0-fast-generate-001"),
    api_key: str = Depends(_require_google_api_key),
    client=Depends(get_async_client),
):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image upload")
    b64 = base64.b64encode(data).decode("ascii")
    mime = image.content_type or "image/png"
   
    messages: list[ChatMessage] = []
    if _rules_text:
        messages.append(ChatMessage(role="system", content=_rules_text))
    messages.append(
        ChatMessage(
            role="user",
            content=[
                TextPart(type="text", text=prompt),
                ImageBase64Part(type="image_base64", data=b64, mime_type=mime),
            ],
        )
    )
    req = ChatRequest(model=model, messages=messages, stream=False)
    p = GeminiProvider(client)
    out = await p.chat(api_key, req)
    parts = out.get("content_parts") or []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "image_base64":
            pmime = part.get("mime_type") or "image/png"
            pdata = part.get("data")
            if isinstance(pdata, str):
                return Response(content=base64.b64decode(pdata), media_type=pmime)
    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No image returned")

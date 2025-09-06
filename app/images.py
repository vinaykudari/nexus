from __future__ import annotations

import base64
from pathlib import Path
import re
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Header, status, Path as FPath
from fastapi.responses import Response

from .deps import get_async_client
from .models import ChatMessage, ChatRequest, ImageBase64Part, TextPart
from .clients.gemini_client import GeminiProvider
from .clients.veo_client import VeoClient
from .config import settings


router = APIRouter(prefix="/v1/images", tags=["images"])
# Public, non-versioned image fetcher as requested: /images/<request-id>/<image-id>
public_router = APIRouter(prefix="/images", tags=["images"])

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
    request_id: str = Form(..., description="Client-supplied request identifier for image grouping"),
    aspect_ratio: Optional[str] = Form(None, alias="aspectRatio", description="Desired aspect ratio, e.g. '16:9' or '1:1'"),
    number_of_images: int = Form(1, alias="numberOfImages", ge=1, le=8, description="How many images to generate (1-8)"),
    api_key: str = Depends(_require_google_api_key),
    client=Depends(get_async_client),
):
    # Sanitize request_id to avoid path traversal or unsafe characters
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id or ""):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id format")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image upload")
    b64 = base64.b64encode(data).decode("ascii")
    mime = image.content_type or "image/png"

    # Build Google Gemini generateContent payload directly to include image generation config
    # such as numberOfImages and aspectRatio.
    contents: list[dict] = []
    if _rules_text:
        # Gemini's systemInstruction is set separately; keep user content clean
        pass
    # User message combining prompt and input image
    parts: list[dict] = [
        {"text": prompt},
        {"inlineData": {"mimeType": mime, "data": b64}},
    ]
    contents.append({"role": "user", "parts": parts})

    payload: dict = {"contents": contents}
    if _rules_text:
        payload["systemInstruction"] = {"parts": [{"text": _rules_text}]}

    gen_cfg: dict = {"candidateCount": int(number_of_images)}
    if aspect_ratio:
        gen_cfg["aspectRatio"] = aspect_ratio
    if gen_cfg:
        payload["generationConfig"] = gen_cfg

    # Call Gemini API
    url = f"{settings.gemini_base_url}/models/{model}:generateContent"
    params = {"key": api_key}
    resp = await client.post(url, json=payload, params=params)
    resp.raise_for_status()
    data_json = resp.json()

    # Extract inlineData images from all candidates
    images: list[tuple[bytes, str]] = []  # (bytes, mime)
    for cand in (data_json.get("candidates") or []):
        content = cand.get("content") or {}
        for p in (content.get("parts") or []):
            inline = p.get("inlineData") if isinstance(p, dict) else None
            if isinstance(inline, dict):
                pmime = inline.get("mimeType") or "image/png"
                pdata = inline.get("data")
                if isinstance(pdata, str):
                    try:
                        images.append((base64.b64decode(pdata), pmime))
                    except Exception:
                        continue

    if not images:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No image returned")

    # Persist images under assets/images/<request_id>/<n>.<ext>
    root = Path(__file__).resolve().parents[1]  # repo root
    out_dir = root / "assets" / "images" / request_id
    
    # Clear existing images for this request_id to ensure clean replacement
    if out_dir.exists():
        for existing_file in out_dir.glob("*"):
            if existing_file.is_file():
                existing_file.unlink()
    
    out_dir.mkdir(parents=True, exist_ok=True)

    def _ext_for_mime(m: str) -> str:
        m = (m or "image/png").lower()
        if m in ("image/jpeg", "image/jpg"):
            return ".jpg"
        if m == "image/webp":
            return ".webp"
        if m == "image/gif":
            return ".gif"
        return ".png"

    for idx, (ibytes, imime) in enumerate(images, start=1):
        ext = _ext_for_mime(imime)
        (out_dir / f"{idx}{ext}").write_bytes(ibytes)

    # Return the first image immediately
    first_bytes, first_mime = images[0]
    return Response(content=first_bytes, media_type=first_mime)


@public_router.get("/{request_id}/{image_id}")
async def get_image(
    request_id: str = FPath(..., description="Request identifier used during generation"),
    image_id: int = FPath(..., ge=1, description="Image index starting at 1"),
):
    # Sanitize request_id
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id or ""):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id format")

    root = Path(__file__).resolve().parents[1]
    base = root / "assets" / "images" / request_id
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")

    # Try known extensions
    exts = [".png", ".jpg", ".jpeg", ".webp", ".gif"]
    chosen_path: Optional[Path] = None
    for ext in exts:
        p = base / f"{image_id}{ext}"
        if p.exists() and p.is_file():
            chosen_path = p
            break
    if chosen_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    ext = chosen_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")
    return Response(content=chosen_path.read_bytes(), media_type=mime)

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
import re
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Header, status, Path as FPath, BackgroundTasks
from fastapi.responses import Response, JSONResponse

from .deps import get_async_client
from .models import ChatMessage, ChatRequest, ImageBase64Part, TextPart
from .clients.gemini_client import GeminiProvider
from .clients.veo_client import VeoClient
from .config import settings
from .prompts import VIBE_CODING_SYSTEM_PROMPT


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
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    image: UploadFile = File(...),
    model: str = Form("gemini-2.5-flash-image-preview"),
    video_model: str = Form("veo-3.0-fast-generate-001"),
    request_id: str = Form(..., description="Client-supplied request identifier for image grouping"),
    aspect_ratio: Optional[str] = Form(None, alias="aspectRatio", description="Desired aspect ratio, e.g. '16:9' or '1:1'"),
    number_of_images: int = Form(1, alias="numberOfImages", ge=1, le=3, description="How many images to generate (1-8)"),
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

    gen_cfg: dict = {}
    if aspect_ratio:
        gen_cfg["aspectRatio"] = aspect_ratio
    if gen_cfg:
        payload["generationConfig"] = gen_cfg

    # Generate first image immediately
    url = f"{settings.gemini_base_url}/models/{model}:generateContent"
    params = {"key": api_key}
    logging.info(f"Requesting first image from Gemini API")
    resp = await client.post(url, json=payload, params=params)
    resp.raise_for_status()
    data_json = resp.json()
    
    # Extract first image
    first_image = None
    candidates = data_json.get("candidates") or []
    for cand in candidates:
        content = cand.get("content") or {}
        for p in (content.get("parts") or []):
            inline = p.get("inlineData") if isinstance(p, dict) else None
            if isinstance(inline, dict):
                pmime = inline.get("mimeType") or "image/png"
                pdata = inline.get("data")
                if isinstance(pdata, str):
                    try:
                        first_image = (base64.b64decode(pdata), pmime)
                        break
                    except Exception:
                        continue
        if first_image:
            break
    
    if not first_image:
        logging.error(f"No image extracted from API response. Full response: {data_json}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No image returned")
    
    logging.info(f"Successfully extracted first image")

    # Setup output directory
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

    # Save original image as "original.{ext}"
    original_ext = _ext_for_mime(mime)
    original_path = out_dir / f"original{original_ext}"
    original_path.write_bytes(data)
    logging.info(f"Saved original image to {original_path} ({len(data)} bytes, {mime})")
    
    # Save first generated image immediately
    first_bytes, first_mime = first_image
    ext = _ext_for_mime(first_mime)
    file_path = out_dir / f"1{ext}"
    file_path.write_bytes(first_bytes)
    logging.info(f"Saved first generated image to {file_path} ({len(first_bytes)} bytes, {first_mime})")
    
    # If more images requested, generate them in background
    if number_of_images > 1:
        background_tasks.add_task(
            _generate_additional_images,
            number_of_images - 1,
            payload,
            url,
            params,
            out_dir,
            _ext_for_mime,
            client
        )
        logging.info(f"Scheduled {number_of_images - 1} additional images for background generation")

    # Return the first image immediately
    return Response(content=first_bytes, media_type=first_mime)


@router.post("/prompt")
async def generate_prompt(
    request_id: str = Form(..., description="Request ID of already generated images"),
    image_id: int = Form(1, description="Image ID to analyze (defaults to 1)"),
    api_key: str = Depends(_require_google_api_key),
    client=Depends(get_async_client),
):
    """Generate a prompt for vibe coding based on a previously generated image."""
    
    # Sanitize request_id
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id or ""):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id format")
    
    # Find the images directory
    root = Path(__file__).resolve().parents[1]
    base = root / "assets" / "images" / request_id
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")

    # Find original image
    exts = [".png", ".jpg", ".jpeg", ".webp", ".gif"]
    original_path: Optional[Path] = None
    for ext in exts:
        p = base / f"original{ext}"
        if p.exists() and p.is_file():
            original_path = p
            break
    if original_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original image not found")
    
    # Find generated image
    generated_path: Optional[Path] = None
    for ext in exts:
        p = base / f"{image_id}{ext}"
        if p.exists() and p.is_file():
            generated_path = p
            break
    if generated_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generated image not found")
    
    # Read and encode both images
    original_data = original_path.read_bytes()
    original_b64 = base64.b64encode(original_data).decode("ascii")
    original_ext = original_path.suffix.lower()
    
    generated_data = generated_path.read_bytes()
    generated_b64 = base64.b64encode(generated_data).decode("ascii")
    generated_ext = generated_path.suffix.lower()
    
    # Determine MIME types
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    original_mime = mime_map.get(original_ext, "image/png")
    generated_mime = mime_map.get(generated_ext, "image/png")
    
    # Use system prompt from prompts.py
    
    # Build user message with both images
    user_parts = [
        {"text": "Here is the original image (reference):"},
        {"inlineData": {"mimeType": original_mime, "data": original_b64}},
        {"text": "Here is the generated/updated image:"},
        {"inlineData": {"mimeType": generated_mime, "data": generated_b64}},
        {"text": "Analyze both images and provide implementation instructions and changes made. If any custom assets are needed (icons, graphics), generate them as actual images - do not include SVG code or asset descriptions in the text response."},
    ]
    
    payload = {
        "contents": [{"role": "user", "parts": user_parts}],
        "systemInstruction": {"parts": [{"text": VIBE_CODING_SYSTEM_PROMPT}]},
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096
        }
    }
    
    # Use Gemini 2.5 Flash Image for image generation capability
    model = "gemini-2.5-flash-image-preview"
    url = f"{settings.gemini_base_url}/models/{model}:generateContent"
    params = {"key": api_key}
    
    logging.info(f"Generating prompt using {model}")
    resp = await client.post(url, json=payload, params=params)
    resp.raise_for_status()
    data_json = resp.json()
    
    # Extract both text and images from response
    generated_text = None
    generated_assets = []
    
    candidates = data_json.get("candidates") or []
    for cand in candidates:
        content = cand.get("content") or {}
        for part in (content.get("parts") or []):
            if "text" in part:
                generated_text = part["text"]
            elif "inlineData" in part:
                # Extract generated asset images
                inline_data = part["inlineData"]
                asset_mime = inline_data.get("mimeType", "image/png")
                asset_data = inline_data.get("data")
                if asset_data:
                    try:
                        asset_bytes = base64.b64decode(asset_data)
                        generated_assets.append({
                            "data": asset_data,
                            "mimeType": asset_mime,
                            "size": len(asset_bytes)
                        })
                    except Exception as e:
                        logging.warning(f"Failed to decode asset: {e}")
    
    if not generated_text:
        logging.error(f"No text generated from API response: {data_json}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No instructions generated")
    
    # Save generated assets to the request directory and prepare for response
    response_assets = []
    if generated_assets:
        assets_dir = base / "assets"
        
        # Clear existing assets for this request_id to ensure clean replacement
        if assets_dir.exists():
            for existing_file in assets_dir.glob("*"):
                if existing_file.is_file():
                    existing_file.unlink()
        
        assets_dir.mkdir(exist_ok=True)
        
        for idx, asset in enumerate(generated_assets, start=1):
            # Determine file extension from MIME type
            ext_map = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
                "image/gif": ".gif",
                "image/svg+xml": ".svg"
            }
            ext = ext_map.get(asset["mimeType"], ".png")
            asset_path = assets_dir / f"asset_{idx}{ext}"
            
            try:
                asset_bytes = base64.b64decode(asset["data"])
                asset_path.write_bytes(asset_bytes)
                
                # Include both asset data and path in response
                response_assets.append({
                    "filename": f"asset_{idx}{ext}",
                    "data": asset["data"],  # Base64 encoded data
                    "path": str(asset_path),  # Full file path
                    "mimeType": asset["mimeType"],
                    "size": len(asset_bytes)
                })
                logging.info(f"Saved generated asset to {asset_path}")
            except Exception as e:
                logging.error(f"Failed to save asset {idx}: {e}")
    
    logging.info(f"Generated instructions ({len(generated_text)} characters) with {len(response_assets)} assets")
    
    return JSONResponse(content={
        "instructions": generated_text,
        "model": model,
        "assets": response_assets
    })


async def _generate_additional_images(
    count: int,
    payload: dict,
    url: str,
    params: dict,
    out_dir: Path,
    ext_for_mime_func,
    client
):
    """Generate additional images in background and save them to disk."""
    try:
        # Create concurrent tasks for remaining images
        tasks = []
        for i in range(count):
            task = asyncio.create_task(_generate_single_image(payload, url, params, client))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Save successful results
        saved_count = 0
        for i, result in enumerate(results, start=2):  # Start from 2 since 1 is already saved
            if isinstance(result, Exception):
                logging.error(f"Failed to generate image {i}: {result}")
                continue
            
            if result:
                image_bytes, mime_type = result
                ext = ext_for_mime_func(mime_type)
                file_path = out_dir / f"{i}{ext}"
                file_path.write_bytes(image_bytes)
                saved_count += 1
                logging.info(f"Saved background image {i} to {file_path} ({len(image_bytes)} bytes, {mime_type})")
        
        logging.info(f"Background generation completed: {saved_count}/{count} additional images saved")
        
    except Exception as e:
        logging.error(f"Error in background image generation: {e}")


async def _generate_single_image(payload: dict, url: str, params: dict, client) -> tuple[bytes, str] | None:
    """Generate a single image and return the bytes and mime type."""
    try:
        resp = await client.post(url, json=payload, params=params)
        resp.raise_for_status()
        data_json = resp.json()
        
        # Extract image from response
        candidates = data_json.get("candidates") or []
        for cand in candidates:
            content = cand.get("content") or {}
            for p in (content.get("parts") or []):
                inline = p.get("inlineData") if isinstance(p, dict) else None
                if isinstance(inline, dict):
                    pmime = inline.get("mimeType") or "image/png"
                    pdata = inline.get("data")
                    if isinstance(pdata, str):
                        try:
                            return (base64.b64decode(pdata), pmime)
                        except Exception:
                            continue
        return None
    except Exception as e:
        logging.error(f"Failed to generate single image: {e}")
        return None


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

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
import os
from pathlib import Path
import re
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Header, status, Path as FPath
from fastapi.responses import Response, JSONResponse
from google import genai

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
    prompt: str = Form(...),
    image: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None),
    model: str = Form("gemini-2.5-flash-image-preview"),
    request_id: str = Form(..., description="Client-supplied request identifier for image grouping"),
    aspect_ratio: Optional[str] = Form(None, alias="aspectRatio", description="Desired aspect ratio, e.g. '16:9' or '1:1'"),
    api_key: str = Depends(_require_google_api_key),
):
    """Apply modifications to an image using AI generation with optional reference image."""
    
    # Log request parameters (excluding sensitive data)
    logging.info(f"/apply endpoint called with parameters: prompt='{prompt}', model={model}, request_id={request_id}")
    
    # Sanitize request_id to avoid path traversal or unsafe characters
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id or ""):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id format")
    
    # Validate image file
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPEG, PNG, WEBP, etc.)"
        )
    
    # Initialize cleanup variables
    temp_files_to_clean = []
    uploaded_files_to_clean = []
    genai_client = None
    
    try:
        # Read and validate main image data
        try:
            main_image_data = await image.read()
            if not main_image_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty image file"
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read image file: {str(e)}"
            )
        
        # Initialize Google genai client
        genai_client = genai.Client(api_key=api_key)
        
        # Create temporary file for main image
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image.filename.split('.')[-1] if image.filename else 'png'}") as temp_file:
            temp_file.write(main_image_data)
            main_image_temp_path = temp_file.name
        temp_files_to_clean.append(main_image_temp_path)
        
        # Upload main image to Google File API
        logging.info("Uploading main image file to Google File API")
        uploaded_main_image = genai_client.files.upload(file=main_image_temp_path)
        uploaded_files_to_clean.append(uploaded_main_image)
        
        # Construct content parts starting with rules + prompt and main image
        enhanced_prompt = f"{_rules_text}\n\n{prompt}" if _rules_text else prompt
        content_parts = [enhanced_prompt, uploaded_main_image]
        
        # Process reference image if provided
        if reference_image:
            try:
                ref_image_data = await reference_image.read()
                if ref_image_data:
                    # Validate reference image
                    if not reference_image.content_type or not reference_image.content_type.startswith('image/'):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Reference file must be an image (JPEG, PNG, WEBP, etc.)"
                        )
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{reference_image.filename.split('.')[-1] if reference_image.filename else 'png'}") as temp_file:
                        temp_file.write(ref_image_data)
                        ref_image_temp_path = temp_file.name
                    temp_files_to_clean.append(ref_image_temp_path)
                    
                    logging.info("Uploading reference image file to Google File API")
                    uploaded_ref_image = genai_client.files.upload(file=ref_image_temp_path)
                    uploaded_files_to_clean.append(uploaded_ref_image)
                    
                    # Insert reference image into content parts
                    content_parts.insert(1, "Use this image as a reference:")
                    content_parts.insert(2, uploaded_ref_image)
            except Exception as e:
                logging.warning(f"Failed to process reference image: {e}")
        
        # Handle aspect ratio validation and prompt modification
        generation_config = {}
        if aspect_ratio:
            supported_ratios = ["1:1", "16:9", "9:16"]
            if aspect_ratio not in supported_ratios:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported aspectRatio: '{aspect_ratio}'. Supported values are {', '.join(supported_ratios)}."
                )
            # Modify enhanced prompt to include aspect ratio instruction
            content_parts[0] = f"{enhanced_prompt} in {aspect_ratio} aspect ratio"
            logging.info(f"Applying aspect ratio: {aspect_ratio} via prompt modification")
        
        # Generate content using Google Generative AI
        logging.info(f"Generating image using {model}")
        
        response = genai_client.models.generate_content(
            model=model,
            contents=content_parts
        )
        
        if not response.candidates:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No response candidates generated"
            )
        
        # Helper function for file extensions
        def _ext_for_mime(m: str) -> str:
            m = (m or "image/png").lower()
            if m in ("image/jpeg", "image/jpg"):
                return ".jpg"
            if m == "image/webp":
                return ".webp"
            if m == "image/gif":
                return ".gif"
            return ".png"
        
        # Setup output directory
        root = Path(__file__).resolve().parents[1]
        out_dir = root / "assets" / "images" / request_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing files for this request_id to ensure clean replacement
        if out_dir.exists():
            for existing_file in out_dir.glob("*"):
                if existing_file.is_file():
                    existing_file.unlink()
        
        # Save original image
        original_ext = _ext_for_mime(image.content_type)
        original_path = out_dir / f"original{original_ext}"
        original_path.write_bytes(main_image_data)
        logging.info(f"Saved original image to {original_path}")
        
        # Process first candidate and extract image
        first_image_bytes, first_image_mime = None, None
        candidate = response.candidates[0]
        
        if candidate.content and candidate.content.parts:
            # Iterate through all parts to find the image
            for part in candidate.content.parts:
                if hasattr(part, 'file_data') and part.file_data and hasattr(part.file_data, 'file_uri'):
                    # Handle FileData response
                    file_response = genai_client.files.get(name=part.file_data.file_uri.split('/')[-1])
                    temp_image_path = f"/tmp/{file_response.name}"
                    file_response.download(file=temp_image_path)
                    with open(temp_image_path, "rb") as f:
                        first_image_bytes = f.read()
                    first_image_mime = file_response.mime_type
                    os.unlink(temp_image_path)  # Clean up temp download
                    break
                elif hasattr(part, 'inline_data') and part.inline_data:
                    # Handle inline data response (more common format)
                    first_image_bytes = part.inline_data.data
                    first_image_mime = part.inline_data.mime_type
                    break
                elif hasattr(part, 'blob') and part.blob and hasattr(part.blob, 'data'):
                    # Handle inline Blob response (fallback)
                    first_image_bytes = part.blob.data
                    first_image_mime = part.blob.mime_type
                    break
        
        if not first_image_bytes:
            logging.error(f"No image extracted from API response. Full response: {response}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No image returned from generation"
            )
        
        # Save generated image
        ext = _ext_for_mime(first_image_mime)
        file_path = out_dir / f"1{ext}"
        file_path.write_bytes(first_image_bytes)
        logging.info(f"Saved generated image to {file_path} ({len(first_image_bytes)} bytes)")
        
        # Return the generated image
        return Response(content=first_image_bytes, media_type=first_image_mime)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logging.error(f"Error during image generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary files
        for path in temp_files_to_clean:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logging.warning(f"Failed to clean up temp file {path}: {e}")
        
        # Clean up uploaded files from Google
        if genai_client:
            for uploaded_file in uploaded_files_to_clean:
                try:
                    genai_client.files.delete(name=uploaded_file.name)
                except Exception as e:
                    logging.warning(f"Failed to clean up uploaded file {uploaded_file.name}: {e}")

@router.post("/prompt")
async def generate_prompt(
    request_id: str = Form(..., description="Request ID of already generated images"),
    image_id: int = Form(1, description="Image ID to analyze (defaults to 1)"),
    model: str = Form("gemini-2.5-flash-image-preview", description="Model to use for prompt generation"),
    api_key: str = Depends(_require_google_api_key),
    client=Depends(get_async_client),
):
    """Generate a prompt for vibe coding based on a previously generated image."""
    
    # Log request parameters (excluding sensitive data)
    logging.info(f"/prompt endpoint called with parameters: request_id={request_id}, image_id={image_id}")
    
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



@router.post("/to-html")
async def image_to_html(
    image: UploadFile = File(...),
    request_id: str = Form(..., description="Client-supplied request identifier for HTML file grouping"),
    model: str = Form("gemini-2.5-pro", description="AI model to use for HTML generation"),
    api_key: str = Depends(_require_google_api_key),
):
    """Convert an image to HTML that looks exactly like the image using Google Generative AI."""
    
    # Sanitize request_id to avoid path traversal or unsafe characters
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id or ""):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id format")

    # Validate image file
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPEG, PNG, WEBP, etc.)"
        )

    # Read and validate image data
    try:
        data = await image.read()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image file: {str(e)}"
        )

    # Create the prompt for HTML generation
    html_generation_prompt = """
Analyze this image and generate complete, production-ready HTML and CSS code that recreates the design shown.

Requirements:
1. Generate semantic HTML5 structure with embedded CSS in <style> tags
2. Match colors, fonts, layout, spacing, and visual elements exactly
3. Use modern CSS techniques (flexbox, grid, etc.) for responsive design
4. Include all text content visible in the image
5. Recreate any buttons, forms, images, or interactive elements as static HTML
6. Use appropriate semantic HTML elements
7. Ensure the result is a complete, standalone HTML file that can be opened in a browser
8. Add subtle hover effects and micro-interactions where appropriate
9. Ensure accessibility (proper alt texts, ARIA labels, semantic markup)
10. Use clean, well-commented code

Return ONLY the complete HTML document with embedded CSS.
Do not include any explanations, markdown formatting, or code blocks - just the raw HTML.
"""

    # Create temporary file for Google File API
    temp_file_path = None
    uploaded_file = None
    
    try:
        # Create temporary file for Google File API upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image.filename.split('.')[-1] if image.filename else 'jpg'}") as temp_file:
            temp_file.write(data)
            temp_file_path = temp_file.name
        
        # Initialize Google genai client
        client = genai.Client(api_key=api_key)
        
        # Upload file to Google File API
        logging.info(f"Uploading image file to Google File API")
        uploaded_file = client.files.upload(file=temp_file_path)
        
        logging.info(f"Generating HTML from image using {model}")
        
        # Generate content using the uploaded file
        response = client.models.generate_content(
            model=model,
            contents=[uploaded_file, html_generation_prompt]
        )
        
        if not response.text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Empty response from the model"
            )
        
        # Clean up the generated HTML (remove markdown code blocks if present)
        generated_html = response.text.strip()
        lines = generated_html.split('\n')
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        
        html_content = '\n'.join(lines).strip()
        
        # Setup output directory
        root = Path(__file__).resolve().parents[1]  # repo root
        html_dir = root / "assets" / "html" / request_id
        html_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing HTML files for this request_id to ensure clean replacement
        if html_dir.exists():
            for existing_file in html_dir.glob("*.html"):
                if existing_file.is_file():
                    existing_file.unlink()
        
        # Save the HTML file
        html_filename = f"generated_{request_id}.html"
        html_path = html_dir / html_filename
        html_path.write_text(html_content, encoding="utf-8")
        
        logging.info(f"Generated HTML saved to {html_path} ({len(html_content)} characters)")
        
        # Return the file path and some metadata
        return JSONResponse(content={
            "html_path": f"{settings.base_url}/html/{request_id}",
            "file_path": str(html_path),
            "relative_path": f"assets/html/{request_id}/{html_filename}",
            "filename": html_filename,
            "size": len(html_content),
            "model": model,
            "message": "HTML file generated successfully. You can open it in a browser to view the result.",
            "view_url": f"/html/{request_id}"
        })
        
    except Exception as e:
        logging.error(f"Error during HTML generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"HTML generation failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logging.warning(f"Failed to clean up temp file: {e}")
        
        # Clean up uploaded file from Google
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                logging.warning(f"Failed to clean up uploaded file: {e}")


# Public HTML router for serving generated HTML files
html_router = APIRouter(prefix="/html", tags=["html"])

@html_router.get("/{request_id}")
async def get_html(
    request_id: str = FPath(..., description="Request identifier used during HTML generation"),
):
    """Serve the generated HTML file for viewing in browser."""
    # Sanitize request_id
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id or ""):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id format")

    root = Path(__file__).resolve().parents[1]
    html_dir = root / "assets" / "html" / request_id
    if not html_dir.exists() or not html_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="HTML request not found")

    # Look for the generated HTML file
    html_filename = f"generated_{request_id}.html"
    html_path = html_dir / html_filename
    
    if not html_path.exists() or not html_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="HTML file not found")

    html_content = html_path.read_text(encoding="utf-8")
    return Response(content=html_content, media_type="text/html")


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

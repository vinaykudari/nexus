from __future__ import annotations

import base64
import logging
import tempfile
import os
from pathlib import Path as PathLib
from typing import Dict, Any, Literal
from enum import Enum

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from google import genai

from .config import settings

# Pydantic models for Swagger documentation
class CodeType(str, Enum):
    html = "html"
    react = "react" 
    vue = "vue"

class CodeGenerationResponse(BaseModel):
    success: bool = Field(..., description="Whether the code generation was successful")
    code: str = Field(..., description="The generated code")
    code_type: CodeType = Field(..., description="Type of code generated")
    model: str = Field(..., description="The AI model used for generation")
    message: str = Field(..., description="Status message")
    request_id: str = Field(..., description="Request ID to access the generated file")
    web_url: str = Field(..., description="URL to view the generated HTML file")
    stats: Dict[str, int] = Field(..., description="Statistics about the generated code")

class CodeGenerationStats(BaseModel):
    code_length: int = Field(..., description="Length of generated code in characters")
    line_count: int = Field(..., description="Number of lines in generated code")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    supported_models: list[str] = Field(..., description="List of supported AI models")
    supported_types: list[str] = Field(..., description="List of supported code generation types")

router = APIRouter(
    prefix="/v1/codegen", 
    tags=["Code Generation"],
    responses={
        400: {"description": "Bad Request - Invalid image file or parameters"},
        401: {"description": "Unauthorized - Missing or invalid API key"},
        422: {"description": "Unprocessable Entity - AI model returned empty response"},
        500: {"description": "Internal Server Error"},
        503: {"description": "Service Unavailable - AI service is down"}
    }
)

# Public router for serving generated HTML files
public_router = APIRouter(prefix="/codegen", tags=["Code Generation - Public"])

async def _require_google_api_key() -> str:
    """Get Google API key from settings"""
    if settings.gemini_api_key:
        return settings.gemini_api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, 
        detail="Missing GEMINI_API_KEY environment variable"
    )

@router.post(
    "/generate-from-image",
    response_model=CodeGenerationResponse,
    summary="Generate Code from Image",
    description="""
    Upload an image and generate production-ready code that recreates the design.
    
    **Supported Code Types:**
    - **HTML**: Complete HTML document with embedded CSS
    - **React**: Modern React component with Tailwind CSS
    - **Vue**: Vue 3 component with Composition API
    
    **Supported Image Formats:** JPEG, PNG, WebP, GIF
    **Maximum File Size:** 10MB
    
    **Example Usage:**
    1. Upload a screenshot of a web design
    2. Select the desired code type (HTML, React, or Vue)  
    3. Get production-ready code that recreates the design
    
    The AI will analyze the visual elements, layout, colors, typography, and generate
    semantic, responsive, and accessible code following modern best practices.
    """,
    responses={
        200: {
            "description": "Code generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "code": "<!DOCTYPE html>\n<html lang=\"en\">...",
                        "code_type": "html",
                        "model": "gemini-2.0-flash-exp", 
                        "message": "Successfully generated html code from image",
                        "stats": {
                            "code_length": 2450,
                            "line_count": 87
                        }
                    }
                }
            }
        }
    }
)
async def generate_code_from_image(
    image: UploadFile = File(
        ..., 
        description="Image file (JPEG, PNG, WebP, GIF) up to 10MB",
        example="design-screenshot.png"
    ),
    request_id: str = Form(
        ..., 
        description="Client-supplied request identifier for file organization",
        example="codegen_123456789"
    ),
    model: str = Form(
        "gemini-2.0-flash-exp", 
        description="AI model to use for code generation",
        example="gemini-2.0-flash-exp"
    ),
    prompt_type: CodeType = Form(
        CodeType.html, 
        description="Type of code to generate",
        example="html"
    ),
    api_key: str = Depends(_require_google_api_key),
) -> CodeGenerationResponse:
    
    # Validate request_id to avoid path traversal or unsafe characters
    if not request_id or not request_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="request_id is required")
    
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
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
    
    # Create temporary file for Google File API
    temp_file_path = None
    uploaded_file = None
    
    # Generate specialized prompts based on code type
    prompts = {
        "html": """
        Analyze this image and generate complete, production-ready HTML and CSS code that recreates the design shown.
        
        Requirements:
        - Generate semantic HTML5 structure
        - Include comprehensive CSS styling to match the visual design exactly
        - Make it fully responsive (mobile-first approach)
        - Use modern CSS features (Grid, Flexbox, CSS custom properties)
        - Include proper spacing, colors, typography, and layout
        - Add subtle hover effects and micro-interactions where appropriate
        - Ensure accessibility (proper alt texts, ARIA labels, semantic markup)
        - Use clean, well-commented code
        - Include a proper CSS reset/normalize
        
        Return ONLY the complete HTML document with embedded CSS in <style> tags.
        Do not include any explanations, markdown formatting, or code blocks - just the raw HTML.
        """,
        
        "react": """
        Analyze this image and generate a modern React component with Tailwind CSS that recreates the design shown.
        
        Requirements:
        - Create a functional React component using modern hooks
        - Use Tailwind CSS classes for all styling
        - Make it fully responsive with Tailwind's responsive prefixes
        - Include proper TypeScript types if applicable
        - Add interactive elements with proper state management
        - Use semantic HTML elements
        - Include accessibility features (ARIA attributes, keyboard navigation)
        - Add subtle animations using Tailwind's animation classes
        - Follow React best practices and clean code principles
        
        Return ONLY the React component code without any explanations or markdown formatting.
        Start with the import statements and export the component at the end.
        """,
        
        "vue": """
        Analyze this image and generate a Vue 3 component with scoped styles that recreates the design shown.
        
        Requirements:
        - Create a Vue 3 component using Composition API
        - Use scoped styles with modern CSS
        - Make it fully responsive
        - Include proper reactive data and computed properties
        - Add interactive elements with Vue's reactivity system
        - Use semantic HTML in the template
        - Include accessibility features
        - Add smooth transitions and animations
        - Follow Vue 3 best practices
        
        Return ONLY the Vue component code (.vue file format) without any explanations or markdown formatting.
        Include <template>, <script setup>, and <style scoped> sections.
        """
    }
    
    try:
        # Select appropriate prompt
        prompt = prompts.get(prompt_type, prompts["html"])
        
        # Create temporary file for Google File API upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image.filename.split('.')[-1] if image.filename else 'jpg'}") as temp_file:
            temp_file.write(data)
            temp_file_path = temp_file.name
        
        # Initialize Google genai client
        client = genai.Client(api_key=api_key)
        
        # Upload file to Google File API
        logging.info(f"Uploading image file to Google File API")
        uploaded_file = client.files.upload(file=temp_file_path)
        
        logging.info(f"Generating {prompt_type} code from image using {model}")
        
        # Generate content using the uploaded file
        response = client.models.generate_content(
            model=model,
            contents=[uploaded_file, prompt]
        )
        
        if not response.text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Empty response from the model"
            )
        
        # Clean up the generated code (remove markdown code blocks if present)
        generated_code = response.text.strip()
        lines = generated_code.split('\n')
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        
        cleaned_code = '\n'.join(lines).strip()
        
        logging.info(f"Successfully generated {len(cleaned_code)} characters of {prompt_type} code")
        
        # Setup output directory
        root = PathLib(__file__).resolve().parents[1]  # repo root
        out_dir = root / "assets" / "codegen" / request_id
        
        # Clear existing files for this request_id to ensure clean replacement
        if out_dir.exists():
            for existing_file in out_dir.glob("*"):
                if existing_file.is_file():
                    existing_file.unlink()
        
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save the original image
        image_ext = image.filename.split('.')[-1] if image.filename and '.' in image.filename else 'jpg'
        image_path = out_dir / f"original.{image_ext}"
        image_path.write_bytes(data)
        logging.info(f"Saved original image to {image_path}")

        # Save the generated code file
        if prompt_type == "html":
            code_file = out_dir / "index.html"
        elif prompt_type == "react":
            code_file = out_dir / "component.jsx"
        elif prompt_type == "vue":
            code_file = out_dir / "component.vue"
        else:
            code_file = out_dir / f"code.{prompt_type}"
            
        code_file.write_text(cleaned_code, encoding='utf-8')
        logging.info(f"Saved generated code to {code_file}")

        # Generate web URL based on your domain
        # For HTML files, we can serve them directly
        if prompt_type == "html":
            web_url = f"https://awake-lauraine-vinaykudari-b9455624.koyeb.app/codegen/{request_id}"
        else:
            # For React/Vue, provide a link to view the raw code
            web_url = f"https://awake-lauraine-vinaykudari-b9455624.koyeb.app/codegen/{request_id}/code"
        
        return CodeGenerationResponse(
            success=True,
            code=cleaned_code,
            code_type=prompt_type,
            model=model,
            message=f"Successfully generated {prompt_type} code from image",
            request_id=request_id,
            web_url=web_url,
            stats={
                "code_length": len(cleaned_code),
                "line_count": len(cleaned_code.split('\n'))
            }
        )
        
    except Exception as e:
        logging.error(f"Error during code generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code generation failed: {str(e)}"
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


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Code Generation Service Health",
    description="Check the health status and capabilities of the code generation service",
    responses={
        200: {
            "description": "Service is healthy and operational",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "codegen",
                        "supported_models": ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
                        "supported_types": ["html", "react", "vue"]
                    }
                }
            }
        }
    }
)
async def codegen_health() -> HealthResponse:
    """Health check endpoint for code generation service."""
    return HealthResponse(
        status="healthy",
        service="codegen",
        supported_models=["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
        supported_types=["html", "react", "vue"]
    )


@public_router.get("/{request_id}")
async def serve_html_file(
    request_id: str = Path(..., description="Request identifier for the generated code"),
):
    """Serve generated HTML file directly in the browser."""
    import re
    # Sanitize request_id
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id or ""):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id format")

    root = PathLib(__file__).resolve().parents[1]
    base = root / "assets" / "codegen" / request_id
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")

    # Try to find the HTML file
    html_file = base / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="HTML file not found")

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_file.read_text(encoding='utf-8'))


@public_router.get("/{request_id}/code")
async def serve_code_file(
    request_id: str = Path(..., description="Request identifier for the generated code"),
):
    """Serve generated code file as plain text."""
    import re
    # Sanitize request_id
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id or ""):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request_id format")

    root = PathLib(__file__).resolve().parents[1]
    base = root / "assets" / "codegen" / request_id
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")

    # Try to find code files in order of preference
    possible_files = ["index.html", "component.jsx", "component.vue"]
    code_file = None
    
    for filename in possible_files:
        file_path = base / filename
        if file_path.exists():
            code_file = file_path
            break
    
    if not code_file:
        # Look for any other code files
        code_files = list(base.glob("code.*"))
        if code_files:
            code_file = code_files[0]
    
    if not code_file:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Code file not found")

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=code_file.read_text(encoding='utf-8'))
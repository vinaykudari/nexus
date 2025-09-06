from __future__ import annotations

import base64
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Literal
from enum import Enum

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx

from .config import settings
from .deps import get_async_client

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
    client: httpx.AsyncClient = Depends(get_async_client),
) -> CodeGenerationResponse:
    
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
    
    # Convert image to base64
    b64_image = base64.b64encode(data).decode('utf-8')
    mime_type = image.content_type
    
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
    
    # Select appropriate prompt
    prompt = prompts.get(prompt_type, prompts["html"])
    
    # Build Gemini API request payload
    contents = [{
        "role": "user",
        "parts": [
            {"text": prompt},
            {"inlineData": {"mimeType": mime_type, "data": b64_image}}
        ]
    }]
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 8192,
            "temperature": 0.1,
            "topP": 0.8,
            "topK": 10
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    
    # Make request to Gemini API
    url = f"{settings.gemini_base_url}/models/{model}:generateContent"
    params = {"key": api_key}
    
    logging.info(f"Generating {prompt_type} code from image using {model}")
    
    try:
        response = await client.post(url, json=payload, params=params)
        response.raise_for_status()
        response_data = response.json()
        
        # Extract generated code from response
        generated_code = ""
        candidates = response_data.get("candidates", [])
        
        if not candidates:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No content generated by the model"
            )
        
        for candidate in candidates:
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if "text" in part:
                    generated_code += part["text"]
        
        if not generated_code.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Empty response from the model"
            )
        
        # Clean up the generated code (remove markdown code blocks if present)
        lines = generated_code.strip().split('\n')
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        
        cleaned_code = '\n'.join(lines).strip()
        
        logging.info(f"Successfully generated {len(cleaned_code)} characters of {prompt_type} code")
        
        return CodeGenerationResponse(
            success=True,
            code=cleaned_code,
            code_type=prompt_type,
            model=model,
            message=f"Successfully generated {prompt_type} code from image",
            stats={
                "code_length": len(cleaned_code),
                "line_count": len(cleaned_code.split('\n'))
            }
        )
        
    except httpx.HTTPStatusError as e:
        error_detail = "Unknown API error"
        if e.response:
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = e.response.text or str(e)
        
        logging.error(f"Gemini API error: {error_detail}")
        raise HTTPException(
            status_code=e.response.status_code if e.response else 500,
            detail=f"API error: {error_detail}"
        )
        
    except httpx.RequestError as e:
        logging.error(f"Request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {str(e)}"
        )
        
    except Exception as e:
        logging.error(f"Unexpected error during code generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


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
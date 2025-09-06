from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # HTTP client settings
    http_timeout_seconds: float = Field(30.0, description="Default HTTP timeout")
    http_max_connections: int = Field(100, description="Max pooled connections")
    http_max_keepalive_connections: int = Field(20, description="Max keepalive connections")

    # Upstream base URLs (override via env for self-hosted proxies)
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_version: str = "2023-06-01"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Optional upstream API keys (fallback when header not provided)
    openai_api_key: str | None = None  # env: OPENAI_API_KEY
    anthropic_api_key: str | None = None  # env: ANTHROPIC_API_KEY
    gemini_api_key: str | None = None  # env: GEMINI_API_KEY

    # Base URL for serving generated content
    base_url: str = "http://localhost:9000"  # env: BASE_URL

    # CORS: kept fully open in app.main (no config needed for hackathon)

    class Config:
        env_file = ".env"


settings = Settings()

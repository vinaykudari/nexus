LLM Proxy (FastAPI)

Fast, minimal proxy server for OpenAI, Anthropic, and Gemini. Use an `X-API-Key` header or set a provider-specific env var; choose provider via path; send a simple chat payload. Supports optional streaming passthrough.

Endpoints

- POST `/v1/chat/{provider}`
  - `provider`: `openai`, `anthropic`, or `gemini`
  - Auth: either provide `X-API-Key: <provider_api_key>` header, or set env var `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` (header takes precedence).
  - Body (JSON):
    {
      "model": "gpt-4o-mini",
      "messages": [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Say hi"}
      ],
      "stream": false,
      "max_tokens": 256,
      "temperature": 0.2
    }

Multimodal content

- `messages[].content` accepts either a string or an array of parts:
  - Text part: `{ "type": "text", "text": "Describe this image" }`
  - Image by URL: `{ "type": "image_url", "url": "https://.../photo.jpg", "mime_type": "image/jpeg" }`
  - Image by base64: `{ "type": "image_base64", "data": "<BASE64>", "mime_type": "image/png" }`

- OpenAI: image_url and base64 supported via chat content parts.
- Anthropic: image URL or base64 supported; requires `max_tokens`.
- Gemini: accepts text, inline images. For `image_url` parts, the proxy fetches and inlines as base64 so you don’t need Google file uploads. API key is passed via query param, no OAuth required.

Non-streaming returns a normalized JSON:
  {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "content": "Hello!",
    "content_parts": [
      {"type":"text","text":"Hello!"}
    ],
    "stop_reason": "stop"
  }

When `stream=true`, the response is Server-Sent Events (`text/event-stream`) with upstream chunks passed through as-is.

Quickstart (uv)

- Python 3.11+
- Install uv: macOS/Linux `curl -LsSf https://astral.sh/uv/install.sh | sh`, Windows `powershell -c "iwr https://astral.sh/uv/install.ps1 -useb | iex"`
- Create env + install deps: `uv sync` (creates `.venv` and resolves `pyproject.toml`)
- Run dev server: `uv run -s dev`
- Run prod server: `uv run -s start`
- Health check: `GET /healthz` → `{ "status": "ok" }`

Examples

OpenAI (non-stream):

curl -s \
  -H "Content-Type: application/json" \
  # Omit header if OPENAI_API_KEY is set in the environment
  -H "X-API-Key: $OPENAI_API_KEY" \
  -X POST http://localhost:8000/v1/chat/openai \
  -d '{
    "model":"gpt-4o-mini",
    "messages":[{"role":"user","content":"Hello!"}],
    "stream": false
  }' | jq

OpenAI (stream):

curl -N \
  -H "Content-Type: application/json" \
  # Omit header if OPENAI_API_KEY is set in the environment
  -H "X-API-Key: $OPENAI_API_KEY" \
  -X POST http://localhost:8000/v1/chat/openai \
  -d '{
    "model":"gpt-4o-mini",
    "messages":[{"role":"user","content":"Stream please"}],
    "stream": true
  }'

Anthropic (non-stream):

curl -s \
  -H "Content-Type: application/json" \
  # Omit header if ANTHROPIC_API_KEY is set in the environment
  -H "X-API-Key: $ANTHROPIC_API_KEY" \
  -X POST http://localhost:8000/v1/chat/anthropic \
  -d '{
    "model":"claude-3-5-sonnet-20240620",
    "messages":[{"role":"system","content":"You are helpful"},{"role":"user","content":"Hello!"}],
    "max_tokens": 256,
    "stream": false
  }' | jq

OpenAI multimodal (image URL):

curl -s \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -X POST http://localhost:8000/v1/chat/openai \
  -d '{
    "model":"gpt-4o-mini",
    "messages":[
      {"role":"user","content":[
        {"type":"text","text":"What is in this image?"},
        {"type":"image_url","url":"https://upload.wikimedia.org/wikipedia/commons/3/3c/Shaki_waterfall.jpg"}
      ]}
    ],
    "stream": false
  }' | jq

Gemini multimodal (image URL):

curl -s \
  -H "Content-Type: application/json" \
  # Omit header if GEMINI_API_KEY is set in the environment
  -H "X-API-Key: $GEMINI_API_KEY" \
  -X POST http://localhost:8000/v1/chat/gemini \
  -d '{
    "model":"gemini-1.5-flash",
    "messages":[
      {"role":"system","content":"You are helpful."},
      {"role":"user","content":[
        {"type":"text","text":"Describe this image succinctly."},
        {"type":"image_url","url":"https://upload.wikimedia.org/wikipedia/commons/3/3c/Shaki_waterfall.jpg"}
      ]}
    ],
    "stream": false
  }' | jq

Notes & Design Choices

- Provider as path param: `/v1/chat/{provider}` keeps body stable and allows simple routing and auth header mapping.
- API key header: Prefer `X-API-Key`, or set env var fallback per provider (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). The proxy maps this to provider-specific auth (`Authorization: Bearer` for OpenAI, `x-api-key` for Anthropic, query param `key` for Gemini).
- Efficiency: single shared `httpx.AsyncClient` with connection pooling; streaming uses `StreamingResponse` and passes upstream SSE bytes through.
- Minimal normalization: for non-stream responses, returns a small, common shape `{provider, model, content, stop_reason}`. Raw upstream payloads can be added later if needed.
- Error handling: upstream HTTP errors are surfaced with the same status code and parsed body when available; network errors return `502`.

Configuration

- Edit `app/config.py` or use env vars (loaded from `.env` if present):
  - `HTTP_TIMEOUT_SECONDS`, `HTTP_MAX_CONNECTIONS`, `HTTP_MAX_KEEPALIVE_CONNECTIONS`
  - `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_VERSION`, `GEMINI_BASE_URL`
  - API keys (optional if you pass header): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`

Consuming images

When models return image data (e.g., Gemini), non-stream responses include `content_parts` with `{ "type":"image_base64", "mime_type":"...", "data":"<BASE64>" }` entries. Decode the base64 to get the binary. For example (Python):

```python
import base64
from pathlib import Path

for part in resp_json.get("content_parts", []):
    if part.get("type") == "image_base64":
        Path("output.png").write_bytes(base64.b64decode(part["data"]))
```

Extending Providers

- Add a new class implementing `ChatProvider` in `app/clients/` and wire it in `app/api.py`’s factory.

Alternative (pip)

If you prefer pip, a `requirements.txt` is included.

- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

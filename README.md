Nexus

Fast, minimal LLM proxy for OpenAI, Anthropic, and Gemini. Choose a provider via the path, send a simple chat payload, and optionally stream responses. Use an `X-API-Key` header or provider-specific env vars.

Features

- Multi‑provider: `openai`, `anthropic`, `gemini`
- Simple auth: `X-API-Key` header or env fallback
- Streaming passthrough: Server‑Sent Events, upstream chunks forwarded as‑is
- Multimodal: text, image URLs, inline base64 images
- Normalized non‑stream responses: consistent `{provider, model, content, content_parts, stop_reason}`
- Efficient: shared `httpx.AsyncClient` with pooling

Quickstart (uv)

- Python 3.11+
- Install uv
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows: `powershell -c "iwr https://astral.sh/uv/install.ps1 -useb | iex"`
- Create env and install deps: `uv sync`
- Run dev server: `uv run -s dev`
- Run prod server: `uv run -s start`
- Health: `GET /healthz` → `{ "status": "ok" }`

Alternative (pip)

- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

API

- POST `/v1/chat/{provider}`
  - `provider`: `openai` | `anthropic` | `gemini`
  - Auth: `X-API-Key: <provider_api_key>` header, or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` (header wins)
  - Request body:

```json
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
```

Multimodal Content

- `messages[].content` may be a string or an array of parts:
  - Text: `{ "type": "text", "text": "Describe this image" }`
  - Image by URL: `{ "type": "image_url", "url": "https://.../photo.jpg", "mime_type": "image/jpeg" }`
  - Image by base64: `{ "type": "image_base64", "data": "<BASE64>", "mime_type": "image/png" }`
- Provider notes:
  - OpenAI: supports image URL and base64 via content parts
  - Anthropic: supports image URL/base64; requires `max_tokens`
  - Gemini: accepts text and inline images; for `image_url`, Nexus fetches and inlines as base64 (no Google uploads needed). API key sent via query param, no OAuth

Responses

- Non‑stream: normalized JSON

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "content": "Hello!",
  "content_parts": [{"type":"text","text":"Hello!"}],
  "stop_reason": "stop"
}
```

- Stream: `text/event-stream` (SSE), upstream chunks passed through unchanged

Examples

OpenAI (non‑stream)

```bash
curl -s \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -X POST http://localhost:8000/v1/chat/openai \
  -d '{
    "model":"gpt-4o-mini",
    "messages":[{"role":"user","content":"Hello!"}],
    "stream": false
  }' | jq
```

OpenAI (stream)

```bash
curl -N \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -X POST http://localhost:8000/v1/chat/openai \
  -d '{
    "model":"gpt-4o-mini",
    "messages":[{"role":"user","content":"Stream please"}],
    "stream": true
  }'
```

Anthropic (non‑stream)

```bash
curl -s \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ANTHROPIC_API_KEY" \
  -X POST http://localhost:8000/v1/chat/anthropic \
  -d '{
    "model":"claude-3-5-sonnet-20240620",
    "messages":[{"role":"system","content":"You are helpful"},{"role":"user","content":"Hello!"}],
    "max_tokens": 256,
    "stream": false
  }' | jq
```

OpenAI multimodal (image URL)

```bash
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
```

Gemini multimodal (image URL)

```bash
curl -s \
  -H "Content-Type: application/json" \
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
```

Configuration

- Edit `app/config.py` or use env vars (loaded from `.env` if present):
  - `HTTP_TIMEOUT_SECONDS`, `HTTP_MAX_CONNECTIONS`, `HTTP_MAX_KEEPALIVE_CONNECTIONS`
  - `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_VERSION`, `GEMINI_BASE_URL`
  - API keys (optional if you pass header): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`

Consuming Images

When models return image data (e.g., Gemini), non‑stream responses include `content_parts` entries like `{ "type":"image_base64", "mime_type":"...", "data":"<BASE64>" }`. Example (Python):

```python
import base64
from pathlib import Path

for part in resp_json.get("content_parts", []):
    if part.get("type") == "image_base64":
        Path("output.png").write_bytes(base64.b64decode(part["data"]))
```

Design Notes

- Provider as path param: stable body, simple routing and auth mapping
- Auth mapping: `X-API-Key` → provider auth (`Authorization: Bearer` for OpenAI, `x-api-key` for Anthropic, `key` query param for Gemini)
- Streaming: FastAPI `StreamingResponse`; forwards upstream SSE bytes untouched
- Errors: upstream HTTP errors bubbled with original status; network errors return `502`

Extending Providers

- Implement `ChatProvider` in `app/clients/` and register via the factory in `app/api.py`.

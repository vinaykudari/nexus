LLM Proxy (FastAPI)

Fast, minimal proxy server for OpenAI and Anthropic. Pass the upstream API key via header, choose provider via path, and send a simple chat payload. Supports optional streaming passthrough.

Endpoints

- POST `/v1/chat/{provider}`
  - `provider`: `openai` or `anthropic`
  - Headers: `X-API-Key: <provider_api_key>`
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

Non-streaming returns a normalized JSON:
  {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "content": "Hello!",
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
  -H "X-API-Key: $ANTHROPIC_API_KEY" \
  -X POST http://localhost:8000/v1/chat/anthropic \
  -d '{
    "model":"claude-3-5-sonnet-20240620",
    "messages":[{"role":"system","content":"You are helpful"},{"role":"user","content":"Hello!"}],
    "max_tokens": 256,
    "stream": false
  }' | jq

Notes & Design Choices

- Provider as path param: `/v1/chat/{provider}` keeps body stable and allows simple routing and auth header mapping.
- API key header: `X-API-Key` is required and transformed to provider-specific auth (`Authorization: Bearer` for OpenAI, `x-api-key` for Anthropic).
- Efficiency: single shared `httpx.AsyncClient` with connection pooling; streaming uses `StreamingResponse` and passes upstream SSE bytes through.
- Minimal normalization: for non-stream responses, returns a small, common shape `{provider, model, content, stop_reason}`. Raw upstream payloads can be added later if needed.
- Error handling: upstream HTTP errors are surfaced with the same status code and parsed body when available; network errors return `502`.

Configuration

- Edit `app/config.py` or use env vars (loaded from `.env` if present):
  - `HTTP_TIMEOUT_SECONDS`, `HTTP_MAX_CONNECTIONS`, `HTTP_MAX_KEEPALIVE_CONNECTIONS`
  - `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_VERSION`

Extending Providers

- Add a new class implementing `ChatProvider` in `app/clients/` and wire it in `app/api.py`’s factory.

Alternative (pip)

If you prefer pip, a `requirements.txt` is included.

- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

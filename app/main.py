from __future__ import annotations

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

from .api import router as api_router
from .images import router as images_router, public_router as images_public_router
from .codegen import router as codegen_router


class _SkipHealthzAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return "/healthz" not in msg


def create_app() -> FastAPI:
    app = FastAPI(title="Nexus", version="0.1.0")

    logging.getLogger("uvicorn.access").addFilter(_SkipHealthzAccessFilter())

    # CORS for browser-based clients (e.g., pixie on Vercel)
    # Allowed origins are configured via settings (env) with a permissive default
    # For the hackathon, keep CORS fully open
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz", tags=["system"])  # simple liveness
    async def healthz():
        return {"status": "ok"}

    app.include_router(api_router)
    app.include_router(images_router)
    # Public image fetcher: /images/<request-id>/<image-id>
    app.include_router(images_public_router)
    app.include_router(codegen_router)

    @app.exception_handler(httpx.HTTPStatusError)
    async def upstream_status_error_handler(_req, exc: httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else 502
        detail = None
        if exc.response is not None:
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
        return JSONResponse(status_code=status_code, content={"detail": detail or str(exc)})

    @app.exception_handler(httpx.RequestError)
    async def upstream_request_error_handler(_req, exc: httpx.RequestError):
        return JSONResponse(status_code=502, content={"detail": f"Upstream request error: {exc}"})

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", reload=True)

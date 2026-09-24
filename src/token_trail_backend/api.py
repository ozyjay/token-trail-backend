"""Narrow public HTTP boundary."""

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from threading import Event, Lock
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from .config import MAX_BODY_BYTES, Settings
from .inference import InferenceEngine, PromptTooLong, TraceTimedOut
from .schemas import TraceRequest


class RateLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            if key not in self.hits and len(self.hits) >= 10_000:
                expired = [
                    name
                    for name, values in self.hits.items()
                    if not values or now - values[-1] >= 60
                ]
                for name in expired:
                    del self.hits[name]
                if len(self.hits) >= 10_000:
                    return False
            hits = self.hits[key]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


def create_app(
    settings: Settings | None = None, engine: Any | None = None, *, load_model: bool = True
) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = engine or InferenceEngine(settings)
    gate = Lock()
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_model:
            await asyncio.to_thread(engine.load)
        yield
        engine.close()

    app = FastAPI(
        title="Token Trail API",
        version="1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def bound_body(request: Request, call_next):
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    if int(length) > MAX_BODY_BYTES:
                        return error(413, "payload_too_large", "request body is too large")
                except ValueError:
                    return error(400, "invalid_request", "invalid Content-Length")
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_BODY_BYTES:
                    return error(413, "payload_too_large", "request body is too large")
            request._body = bytes(body)
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        if exc.status_code == 400:
            return error(400, "invalid_request", "request body is not valid JSON")
        return error(exc.status_code, "http_error", "request failed")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError):
        return error(422, "invalid_request", "request does not match the trace contract")

    @app.get("/health")
    async def health():
        return {"status": "ready" if engine.ready else "not_ready", "ready": bool(engine.ready)}

    @app.get("/v1/models")
    async def models():
        return {
            "models": [
                {
                    "alias": settings.model_alias,
                    "repository": settings.model_repo,
                    "revision": settings.model_revision,
                    "ready": bool(engine.ready),
                }
            ]
        }

    @app.post("/v1/traces")
    async def traces(payload: TraceRequest, request: Request):
        if not engine.ready:
            return error(503, "not_ready", "model is not ready")
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            return error(429, "rate_limited", "request rate limit exceeded")
        if not gate.acquire(blocking=False):
            return error(429, "busy", "trace worker is busy")
        cancel = Event()

        def run():
            try:
                return engine.trace(payload, cancel=cancel)
            finally:
                gate.release()

        task = asyncio.create_task(asyncio.to_thread(run))
        deadline = time.monotonic() + settings.request_timeout_seconds
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                if await request.is_disconnected():
                    cancel.set()
                    return error(499, "client_disconnected", "client disconnected")
                if time.monotonic() >= deadline:
                    cancel.set()
                    return error(504, "timeout", "trace exceeded time budget")
            return await task
        except PromptTooLong as exc:
            return error(422, "prompt_too_long", str(exc))
        except TraceTimedOut as exc:
            return error(504, "timeout", str(exc))
        except Exception:
            return error(500, "inference_failed", "trace generation failed")
        finally:
            if not task.done():
                cancel.set()

    return app


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


app = create_app()

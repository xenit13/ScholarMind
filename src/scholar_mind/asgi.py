from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from scholar_mind.api.routes.eval import router as eval_router
from scholar_mind.api.routes.health import router as health_router
from scholar_mind.api.routes.sessions import router as sessions_router
from scholar_mind.app import get_container

_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if get_container.cache_info().currsize:
        await get_container().aclose()
        get_container.cache_clear()


def create_app() -> FastAPI:
    app = FastAPI(title="ScholarMind", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(sessions_router)
    app.include_router(eval_router)
    app.include_router(health_router)

    @app.get("/")
    async def serve_index():
        return FileResponse(_STATIC_DIR / "index.html")

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "data": None,
                "error": {"code": "INTERNAL_ERROR", "message": str(exc)},
                "meta": {"request_id": "internal", "timestamp": None, "latency_ms": 0},
            },
        )

    return app


app = create_app()

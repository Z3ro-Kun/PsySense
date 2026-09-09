"""
api/main.py

bridge_api: the read/enroll HTTP layer over psysense's SQLite data,
consumed by the React dashboard. Runs as its own process, on the same
machine as the camera pipeline (see README's "Deployment topology" note),
sharing the same SQLite file via its own AsyncDBWriter/Repository -- this
was anticipated by db/writer.py's docstring, which already names bridge_api
as an expected independent reader/occasional-writer.

Run (see api/asgi.py for the actual entrypoint):
    uvicorn api.asgi:app --host 0.0.0.0 --port 8080
or, for local dev with auto-reload:
    uvicorn api.asgi:app --reload

Deliberately no module-level `app = create_app()` here: importing this
module (e.g. from tests, which want create_app() to build fake-backed
instances) must never eagerly load real ML models or require
ADMIN_*/JWT_SECRET to be set. api/asgi.py is the only place that happens.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.auth import AuthSettings, FailedLoginTracker
from api.routers import auth as auth_router
from api.routers import identity_candidates as identity_candidates_router
from api.routers import students as students_router
from api.routers import telemetry as telemetry_router
from api.routers import users as users_router
from core.config import AppConfig, load_config
from db.repository.repository import Repository
from db.writer import AsyncDBWriter
from logging_.logger import get_logger
from services.identity.embedder import get_embedder
from services.identity.manager import IdentityManager
from services.identity.quality import HeuristicQualityAssessor
from services.identity.vector_index import LinearVectorIndex

logger = get_logger("api.main")


def create_app(
    config: Optional[AppConfig] = None,
    *,
    repository: Optional[Repository] = None,
    writer: Optional[AsyncDBWriter] = None,
    identity_manager: Optional[IdentityManager] = None,
    auth_settings: Optional[AuthSettings] = None,
) -> FastAPI:
    """App factory. All heavy/stateful dependencies are overridable so
    tests can inject fakes (a temp-file repository, a fake embedder-backed
    IdentityManager, env-free AuthSettings) instead of loading real models
    or requiring ADMIN_* environment variables to be set."""
    config = config or load_config()

    if writer is None and repository is None:
        writer = AsyncDBWriter(
            db_path=config.pipeline.database.path,
            schema_path=str(Path(__file__).resolve().parents[1] / "db" / "schema.sql"),
        )
        writer.start()
    if repository is None:
        repository = Repository(writer, config.pipeline.database.path)

    if auth_settings is None:
        auth_settings = AuthSettings()  # fails loudly if ADMIN_*/JWT_SECRET are unset

    if identity_manager is None:
        embedder = get_embedder(
            ctx_id=config.identity.embedder.ctx_id, det_size=tuple(config.identity.embedder.det_size),
        )
        quality = HeuristicQualityAssessor(
            blur_threshold=config.identity.quality.blur_threshold,
            brightness_range=(config.identity.quality.brightness_min, config.identity.quality.brightness_max),
            max_yaw_deg=config.identity.quality.max_yaw_deg,
            max_occlusion=config.identity.quality.max_occlusion,
            min_face_size_px=config.identity.quality.min_face_size_px,
        )
        identity_manager = IdentityManager(
            embedder=embedder,
            quality_assessor=quality,
            # Throwaway index: this instance only ever calls enroll_student(),
            # never resolve()/search() -- face recognition stays the running
            # pipeline's job, which picks up new enrollments via
            # IdentityManager.refresh_from_repository().
            vector_index=LinearVectorIndex(dim=config.identity.vector_index.dim),
            match_distance_threshold=config.identity.matching.match_distance_threshold,
            duplicate_distance_threshold=config.identity.matching.duplicate_distance_threshold,
            max_embeddings_per_student=config.identity.matching.max_embeddings_per_student,
            repository=repository,
        )
        identity_manager.load_from_repository()

    app = FastAPI(title="PsySense bridge_api", version="1.0.0")

    app.state.config = config
    app.state.repo = repository
    app.state.auth_settings = auth_settings
    app.state.identity_manager = identity_manager
    app.state.login_tracker = FailedLoginTracker(
        max_failures=config.api.max_failed_logins, lockout_seconds=config.api.login_lockout_seconds,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[config.api.frontend_url],
        allow_credentials=False,  # Bearer token in Authorization header, not a cookie
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_exception_handlers(app)

    app.include_router(auth_router.router)
    app.include_router(students_router.router)
    app.include_router(telemetry_router.router)
    app.include_router(identity_candidates_router.router)
    app.include_router(users_router.router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        """Liveness only -- no dependency checks. Always 200 if the
        process is up."""
        return {"status": "ok", "service": "bridge_api"}

    @app.get("/ready", tags=["meta"])
    def ready() -> JSONResponse:
        """Readiness -- verifies the database is actually reachable, not
        just that the process started (CLAUDE.md Sec 9's "started != ready")."""
        try:
            repository.count_students()
        except Exception as exc:  # noqa: BLE001
            logger.error("Readiness check failed: database unreachable", extra={"context": {"error": str(exc)}})
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"error": {"code": "DATABASE_UNAVAILABLE", "message": "Database is not reachable."}},
            )
        return JSONResponse(status_code=200, content={"status": "ready"})

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Every error response -- expected (404/401/429) or not -- goes out
    in the same {"error": {"code", "message"}} envelope, and raw
    tracebacks never reach the client (CLAUDE.md Sec 20)."""

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": _code_for_status(exc.status_code), "message": str(exc.detail)}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": {"code": "VALIDATION_ERROR", "message": "Request did not match the expected schema."}},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception in bridge_api",
            extra={"context": {"path": str(request.url.path), "error": str(exc)}},
        )
        logger.debug(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}},
        )


def _code_for_status(status_code: int) -> str:
    return {
        status.HTTP_400_BAD_REQUEST: "BAD_REQUEST",
        status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
        status.HTTP_404_NOT_FOUND: "NOT_FOUND",
        status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMITED",
    }.get(status_code, "ERROR")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import claims, evaluate, evidence, health, pipeline, verify
from core.config import get_settings
from core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Arabic Fact-Checking API",
        description="Pipeline for retrieving evidence, generating QA pairs, and verifying Arabic claims.",  # noqa: E501
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(claims.router)
    app.include_router(evidence.router)
    app.include_router(verify.router)
    app.include_router(pipeline.router)
    app.include_router(evaluate.router)

    return app


app = create_app()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from backend.app.api import predict, health, feedback
from backend.app.core.config import settings
from backend.app.core.logging import setup_logging
from backend.app.core.metrics import backend_info, model_info
from backend.app.services.model_service import init_model_service

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RadiologyAI starting up...")
    backend_info.labels(
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
    ).set(1)
    model_info.labels(
        model_name=settings.MODEL_NAME,
        framework=settings.MODEL_FRAMEWORK,
        version=settings.APP_VERSION,
    ).set(1)
    init_model_service(settings.MODEL_PATH)

    # Load historical feedback and init Prometheus gauges
    from backend.app.api.feedback import load_feedback_from_file
    load_feedback_from_file()

    logger.info("Model loaded. Ready to serve.")
    yield
    logger.info("RadiologyAI shutting down.")

app = FastAPI(
    title       = settings.APP_TITLE,
    description = settings.APP_DESCRIPTION,
    version     = settings.APP_VERSION,
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

Instrumentator().instrument(app).expose(app)

app.include_router(health.router,   tags=["Health"])
app.include_router(predict.router,  prefix=settings.API_PREFIX, tags=["Predict"])
app.include_router(feedback.router, prefix=settings.API_PREFIX, tags=["Feedback"])
from backend.app.api import reports
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])

logger.info("All routers registered.")

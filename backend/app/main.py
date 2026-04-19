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
from backend.app.services.model_service import init_model_service

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RadiologyAI starting up...")
    init_model_service(settings.MODEL_PATH)
    logger.info("Model loaded. Ready to serve.")
    yield
    logger.info("RadiologyAI shutting down.")

app = FastAPI(
    title       = "RadiologyAI API",
    description = "Chest X-Ray Disease Classifier",
    version     = "1.0.0",
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
app.include_router(predict.router,  prefix="/api/v1", tags=["Predict"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])

logger.info("All routers registered.")

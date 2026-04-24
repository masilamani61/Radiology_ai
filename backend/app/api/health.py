from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas.predict import HealthResponse, ClassesResponse
from backend.app.services.model_service import get_model_service

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health():
    service = get_model_service()
    return HealthResponse(
        status="ok",
        model_loaded=service.is_loaded() if service else False,
        version=settings.APP_VERSION,
        model_name=settings.MODEL_NAME,
        framework=settings.MODEL_FRAMEWORK,
        image_size=settings.IMAGE_SIZE,
        class_count=len(settings.CLASS_NAMES),
    )

@router.get("/ready")
async def ready():
    service = get_model_service()
    if not service or not service.is_loaded():
        return {"status": "not ready", "model_loaded": False}
    return {"status": "ready", "model_loaded": True}

@router.get("/classes", response_model=ClassesResponse)
async def get_classes():
    classes = settings.CLASS_NAMES
    return ClassesResponse(classes=classes, total=len(classes))

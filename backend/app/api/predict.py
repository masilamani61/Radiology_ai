import logging
import uuid
from io import BytesIO

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from backend.app.core.config import settings
from backend.app.core.metrics import (
    inference_latency_ms,
    prediction_confidence,
    prediction_total,
    upload_validation_failures_total,
)
from backend.app.schemas.predict import PredictResponse, ClassProbability
from backend.app.services.model_service import get_model_service

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_TYPES = settings.ALLOWED_UPLOAD_TYPES
MAX_SIZE_MB   = settings.MAX_UPLOAD_SIZE_MB


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    POST /predict
    Input : chest X-ray image (JPEG/PNG, max 10MB)
    Output: predicted class, confidence, risk level, Grad-CAM heatmap
    """
    # validate file type
    if file.content_type not in ALLOWED_TYPES:
        upload_validation_failures_total.labels(reason="invalid_content_type").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type: {file.content_type}. Must be JPEG or PNG."
        )

    contents = await file.read()

    # validate file size
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        upload_validation_failures_total.labels(reason="file_too_large").inc()
        raise HTTPException(
            status_code=422,
            detail=f"File too large. Maximum size is {MAX_SIZE_MB}MB."
        )

    try:
        image = Image.open(BytesIO(contents)).convert("RGB").resize(
            (settings.IMAGE_SIZE, settings.IMAGE_SIZE)
        )
        image_array = np.array(image, dtype=np.uint8)
    except Exception as e:
        upload_validation_failures_total.labels(reason="invalid_image").inc()
        raise HTTPException(status_code=422, detail=f"Invalid image: {e}")

    service = get_model_service()
    if not service or not service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        result = service.predict(
            image_array,
            generate_gradcam=True,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")

    logger.info(
        f"Prediction: {result['predicted_class']} "
        f"confidence={result['confidence']:.3f} "
        f"latency={result['inference_time_ms']:.1f}ms"
    )
    prediction_total.labels(
        predicted_class=result["predicted_class"],
        risk_level=result["risk_level"],
    ).inc()
    prediction_confidence.observe(result["confidence"])
    inference_latency_ms.observe(result["inference_time_ms"])

    return PredictResponse(
        predicted_class   = result["predicted_class"],
        confidence        = result["confidence"],
        risk_level        = result["risk_level"],
        all_probabilities = [ClassProbability(**p) for p in result["all_probabilities"]],
        gradcam_base64    = result["gradcam_base64"],
        inference_time_ms = result["inference_time_ms"],
    )

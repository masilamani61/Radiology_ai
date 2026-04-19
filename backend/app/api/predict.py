import logging
import uuid
from io import BytesIO

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from backend.app.schemas.predict import PredictResponse, ClassProbability
from backend.app.services.model_service import get_model_service

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_SIZE_MB   = 10


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    POST /predict
    Input : chest X-ray image (JPEG/PNG, max 10MB)
    Output: predicted class, confidence, risk level, Grad-CAM heatmap
    """
    # validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type: {file.content_type}. Must be JPEG or PNG."
        )

    contents = await file.read()

    # validate file size
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=422,
            detail=f"File too large. Maximum size is {MAX_SIZE_MB}MB."
        )

    try:
        image = Image.open(BytesIO(contents)).convert("RGB").resize((224, 224))
        image_array = np.array(image, dtype=np.uint8)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid image: {e}")

    service = get_model_service()
    if not service or not service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        result = service.predict(image_array, generate_gradcam=True)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")

    logger.info(
        f"Prediction: {result['predicted_class']} "
        f"confidence={result['confidence']:.3f} "
        f"latency={result['inference_time_ms']:.1f}ms"
    )

    return PredictResponse(
        predicted_class   = result["predicted_class"],
        confidence        = result["confidence"],
        risk_level        = result["risk_level"],
        all_probabilities = [ClassProbability(**p) for p in result["all_probabilities"]],
        gradcam_base64    = result["gradcam_base64"],
        inference_time_ms = result["inference_time_ms"],
    )

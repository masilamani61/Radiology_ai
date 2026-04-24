import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.app.core.config import settings
from backend.app.core.metrics import model_loaded_gauge
from ml.src.models.model import get_model
from ml.src.data.dataloader import get_val_transforms
from ml.src.explainability.gradcam import explain_single

logger = logging.getLogger(__name__)

CLASS_NAMES = settings.CLASS_NAMES
RISK_MAP    = settings.RISK_MAP


class ModelService:
    def __init__(self, model_path: str):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform  = get_val_transforms()
        self.model      = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            self.model = get_model(self.device, freeze=False)
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()
            model_loaded_gauge.set(1)
            logger.info(f"Model loaded from {self.model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.model = None
            model_loaded_gauge.set(0)

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image_array: np.ndarray, generate_gradcam: bool = True):
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        start = time.time()

        # preprocess
        aug    = self.transform(image=image_array)
        tensor = aug["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)[0]

        pred_idx    = probs.argmax().item()
        pred_class  = CLASS_NAMES[pred_idx]
        confidence  = probs[pred_idx].item()
        inf_time_ms = (time.time() - start) * 1000

        # Grad-CAM
        gradcam_b64 = None
        if generate_gradcam:
            try:
                rgb_float   = image_array.astype(np.float32) / 255.0
                gradcam_b64 = explain_single(
                    self.model, tensor, rgb_float, pred_idx, self.device
                )
            except Exception as e:
                logger.warning(f"Grad-CAM failed: {e}")

        all_probs = [
            {"class_name": CLASS_NAMES[i], "confidence": round(probs[i].item(), 4)}
            for i in range(len(CLASS_NAMES))
        ]

        return {
            "predicted_class"  : pred_class,
            "confidence"       : round(confidence, 4),
            "risk_level"       : RISK_MAP[pred_class],
            "all_probabilities": all_probs,
            "gradcam_base64"   : gradcam_b64,
            "inference_time_ms": round(inf_time_ms, 2),
        }


# singleton — loaded once on startup
_service: ModelService = None

def get_model_service() -> ModelService:
    return _service

def init_model_service(model_path: str):
    global _service
    _service = ModelService(model_path)

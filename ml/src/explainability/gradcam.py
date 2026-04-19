"""
ml/src/explainability/gradcam.py
=================================
Grad-CAM heatmap generation for chest X-ray predictions.
Uses pytorch-grad-cam library targeting EfficientNetB0's
last convolutional block.

Two modes:
  1. Batch mode  — generates sample heatmaps for all 3 classes
                   and saves as artefacts (DVC + MLflow)
  2. Single mode — used by the FastAPI backend at inference time

Usage:
    # Batch mode (DVC stage)
    python ml/src/explainability/gradcam.py

    # Single image (called from FastAPI)
    from ml.src.explainability.gradcam import explain_single
    heatmap_b64 = explain_single(model, image_tensor, pred_class)
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional

import cv2
import mlflow
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ml.src.data.dataloader import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    get_val_transforms,
)
from ml.src.models.model import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[3]
MODELS_DIR  = ROOT / "ml" / "models"
PROC_DIR    = ROOT / "data" / "processed"
SAMPLES_DIR = ROOT / "ml" / "experiments" / "gradcam_samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def load_model(device: torch.device) -> torch.nn.Module:
    """
    Load trained EfficientNetB0 weights.

    Args:
        device: torch.device

    Returns:
        Model in eval mode on the given device
    """
    model_path = MODELS_DIR / "efficientnetb0_best.pth"
    model      = get_model(device, freeze=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_gradcam_target_layer(model: torch.nn.Module):
    """
    Return the target convolutional layer for Grad-CAM.
    For EfficientNetB0, the last block of features is used.

    Args:
        model: XRayClassifier instance

    Returns:
        Target layer module
    """
    # EfficientNetB0 last convolutional block
    return [model.backbone.features[-1]]


def preprocess_image(
    image_path : Path,
    device     : torch.device,
) -> tuple:
    """
    Load and preprocess a single image for inference.

    Args:
        image_path: Path to image file
        device:     torch.device

    Returns:
        (input_tensor [1,C,H,W], rgb_float_array [H,W,3])
        rgb_float_array is needed for heatmap overlay
    """
    transform = get_val_transforms()

    pil_img  = Image.open(image_path).convert("RGB").resize((224, 224))
    rgb_float = np.array(pil_img, dtype=np.float32) / 255.0

    aug       = transform(image=np.array(pil_img, dtype=np.uint8))
    tensor    = aug["image"].unsqueeze(0).to(device)    # [1,3,224,224]

    return tensor, rgb_float


def generate_heatmap(
    model       : torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float   : np.ndarray,
    target_class: int,
    device      : torch.device,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a given class.

    Args:
        model:        Trained model
        input_tensor: Preprocessed input [1,C,H,W]
        rgb_float:    Original image as float array [H,W,3]
        target_class: Class index to explain
        device:       torch.device

    Returns:
        Heatmap overlaid on original image [H,W,3] uint8
    """
    target_layers = get_gradcam_target_layer(model)
    targets       = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor = input_tensor,
            targets      = targets,
        )[0]                                        # [H,W] float [0,1]

    # overlay heatmap on original image
    visualization = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb   = True,
        colormap  = cv2.COLORMAP_JET,
        image_weight= 0.5,
    )
    return visualization                            # [H,W,3] uint8


def tensor_to_base64(image_array: np.ndarray) -> str:
    """
    Convert numpy image array to base64 PNG string.
    Used by the FastAPI endpoint to return the heatmap.

    Args:
        image_array: [H,W,3] uint8 numpy array

    Returns:
        Base64-encoded PNG string (without data URI prefix)
    """
    pil_img    = Image.fromarray(image_array)
    buffer     = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def explain_single(
    model       : torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float   : np.ndarray,
    target_class: int,
    device      : torch.device,
) -> str:
    """
    Generate Grad-CAM heatmap for a single inference.
    Called by the FastAPI /explain endpoint.

    Args:
        model:        Trained model (already loaded)
        input_tensor: Preprocessed tensor [1,C,H,W]
        rgb_float:    Original image float array [H,W,3]
        target_class: Predicted class index
        device:       torch.device

    Returns:
        Base64-encoded PNG heatmap string
    """
    heatmap = generate_heatmap(
        model, input_tensor, rgb_float, target_class, device
    )
    return tensor_to_base64(heatmap)


def run_batch_samples(
    model  : torch.nn.Module,
    device : torch.device,
    n_per_class: int = 3,
) -> None:
    """
    Generate and save sample Grad-CAM images for each class.
    Run as part of the DVC 'explain' stage.
    Saves PNGs to ml/experiments/gradcam_samples/ and logs to MLflow.

    Args:
        model:       Trained model
        device:      torch.device
        n_per_class: Number of samples to generate per class
    """
    logger.info("Generating Grad-CAM samples for all classes...")
    saved_paths = []

    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = PROC_DIR / "test" / class_name
        if not class_dir.exists():
            logger.warning(f"Test folder not found: {class_dir}")
            continue

        images = list(class_dir.glob("*.jpg"))[:n_per_class]
        logger.info(f"  {class_name}: generating {len(images)} samples")

        for i, img_path in enumerate(images):
            input_tensor, rgb_float = preprocess_image(img_path, device)

            # generate heatmap for the true class
            heatmap = generate_heatmap(
                model, input_tensor, rgb_float, class_idx, device
            )

            # side-by-side: original | heatmap
            original_uint8 = (rgb_float * 255).astype(np.uint8)
            combined       = np.hstack([original_uint8, heatmap])

            out_path = SAMPLES_DIR / f"{class_name}_sample_{i+1}.png"
            Image.fromarray(combined).save(out_path)
            saved_paths.append(out_path)
            logger.info(f"    Saved: {out_path.name}")

    logger.info(f"\nTotal Grad-CAM samples saved: {len(saved_paths)}")

    # log all samples to MLflow
    try:
        mlflow.set_experiment("radiologyai_xray_classification")
        with mlflow.start_run(run_name="gradcam_samples"):
            for p in saved_paths:
                mlflow.log_artifact(str(p), artifact_path="gradcam_samples")
        logger.info("Grad-CAM samples logged to MLflow.")
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = load_model(device)
    run_batch_samples(model, device, n_per_class=3)
    logger.info("Grad-CAM stage complete.")

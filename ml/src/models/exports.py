"""
ml/src/models/export.py
========================
Export trained EfficientNetB0 to TorchScript for production serving.
TorchScript allows the model to run without the Python source —
required for the FastAPI inference service and TorchServe deployment.

Also registers the final model in the MLflow Model Registry.

Usage:
    python ml/src/models/export.py
"""

import json
import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from ml.src.models.model import get_model
from shared.config import get_config_value, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[3]
MODELS_DIR  = resolve_path(get_config_value("paths", "models_dir", default="ml/models"))
EXP_DIR     = resolve_path(get_config_value("paths", "experiments_dir", default="ml/experiments"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SIZE = get_config_value("ml", "image_size", default=224)
MLFLOW_SERVICE_URL = get_config_value(
    "ops",
    "mlflow",
    "service_url",
    default="http://localhost:5000",
)
MLFLOW_EXPERIMENT_NAME = get_config_value(
    "ops",
    "mlflow",
    "experiment_name",
    default="radiologyai_xray_classification",
)
REGISTERED_MODEL_NAME = get_config_value(
    "ml",
    "model",
    "registered_model_name",
    default="radiologyai_xray_classifier",
)


def export_torchscript(
    model      : torch.nn.Module,
    device     : torch.device,
    save_path  : Path,
) -> None:
    """
    Export model to TorchScript via tracing.

    Args:
        model:     Trained model in eval mode
        device:    torch.device
        save_path: Output .pt file path
    """
    model.eval()

    # dummy input for tracing
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    logger.info("Tracing model to TorchScript...")
    scripted = torch.jit.trace(model, dummy_input)

    torch.jit.save(scripted, save_path)
    logger.info(f"TorchScript model saved: {save_path}")

    # verify it loads and runs
    loaded = torch.jit.load(save_path, map_location=device)
    with torch.no_grad():
        out = loaded(dummy_input)
    logger.info(
        f"Export verified — output shape: {out.shape}, "
        f"file size: {save_path.stat().st_size / 1024 / 1024:.1f} MB"
    )


def register_to_mlflow(
    model      : torch.nn.Module,
    mlflow_uri : str = MLFLOW_SERVICE_URL,
) -> str:
    """
    Register the final model in the MLflow Model Registry.
    Required by evaluation guideline — 'MLflow for APIification'.

    Args:
        model:      Trained PyTorch model
        mlflow_uri: MLflow tracking server URL

    Returns:
        MLflow model URI string
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="model_export") as run:
        model_info = mlflow.pytorch.log_model(
            pytorch_model         = model,
            artifact_path         = "model",
            registered_model_name = REGISTERED_MODEL_NAME,
        )
        model_uri = model_info.model_uri
        logger.info(f"Model registered in MLflow: {model_uri}")
        logger.info(f"MLflow run ID: {run.info.run_id}")
        return model_uri


def main() -> None:
    """Load best checkpoint, export to TorchScript, register to MLflow."""
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_ckpt_path = resolve_path(
        get_config_value("paths", "model_path", default="ml/models/efficientnetb0_best.pth")
    )

    logger.info("=" * 55)
    logger.info("RadiologyAI — Model Export")
    logger.info("=" * 55)
    logger.info(f"Device          : {device}")
    logger.info(f"Checkpoint      : {best_ckpt_path}")

    if not best_ckpt_path.exists():
        logger.error(
            f"Checkpoint not found: {best_ckpt_path}\n"
            f"Run train.py first."
        )
        return

    # ── Load model ────────────────────────────────────────────
    model = get_model(device, freeze=False)
    model.load_state_dict(
        torch.load(best_ckpt_path, map_location=device)
    )
    model.eval()
    logger.info("Checkpoint loaded successfully.")

    # ── Export TorchScript ────────────────────────────────────
    ts_path = resolve_path(
        get_config_value("paths", "torchscript_model_path", default="ml/models/efficientnetb0_torchscript.pt")
    )
    export_torchscript(model, device, ts_path)

    # ── Register to MLflow ────────────────────────────────────
    try:
        model_uri = register_to_mlflow(model)
        logger.info(f"MLflow URI: {model_uri}")
    except Exception as e:
        logger.warning(f"MLflow registration skipped (server not running?): {e}")

    # ── Save export manifest ──────────────────────────────────
    manifest = {
        "checkpoint"   : str(best_ckpt_path),
        "torchscript"  : str(ts_path),
        "model_name"   : get_config_value("ml", "model", "architecture", default="EfficientNetB0"),
        "framework"    : get_config_value("ml", "model", "framework", default="PyTorch 2.3.0"),
        "classes"      : get_config_value("ml", "class_names", default=["Normal", "Pneumonia", "COVID19"]),
        "input_shape"  : [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "export_device": str(device),
    }
    manifest_path = resolve_path(
        get_config_value("paths", "export_manifest_path", default="ml/models/export_manifest.json")
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("\n" + "=" * 55)
    logger.info("Export complete!")
    logger.info(f"  TorchScript : {ts_path}")
    logger.info(f"  Manifest    : {manifest_path}")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()

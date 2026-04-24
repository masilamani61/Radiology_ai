"""
Promotes best MLflow model to Production in the Model Registry.
Shows model versioning and lifecycle management.
"""
import mlflow
from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI  = "http://localhost:5005"
MODEL_NAME  = "radiologyai_xray_classifier"
MIN_F1      = 0.70


def promote_best_model():
    """Find best run by val_f1 and promote to Production."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Search all runs
    runs = client.search_runs(
        experiment_ids = ["1"],
        order_by       = ["metrics.val_f1 DESC"],
        max_results    = 10,
    )

    if not runs:
        logger.error("No runs found!")
        return

    logger.info(f"Found {len(runs)} runs:")
    for r in runs:
        f1 = r.data.metrics.get("val_f1", 0)
        logger.info(f"  Run {r.info.run_id[:8]} | val_f1={f1:.4f}")

    best_run = runs[0]
    best_f1  = best_run.data.metrics.get("val_f1", 0)

    logger.info(f"\nBest run: {best_run.info.run_id[:8]} | val_f1={best_f1:.4f}")

    if best_f1 < MIN_F1:
        logger.warning(f"Best F1 {best_f1:.4f} below threshold {MIN_F1} — not promoting")
        return

    # Register model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    try:
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        logger.info(f"Model registered: version={mv.version}")

        # Transition to Production
        client.transition_model_version_stage(
            name    = MODEL_NAME,
            version = mv.version,
            stage   = "Production",
            archive_existing_versions=True,
        )
        logger.info(f"Model v{mv.version} promoted to Production!")

        # Add description
        client.update_model_version(
            name        = MODEL_NAME,
            version     = mv.version,
            description = f"EfficientNetB0 | val_f1={best_f1:.4f} | auto-promoted",
        )

    except Exception as e:
        logger.error(f"Registration failed: {e}")


if __name__ == "__main__":
    promote_best_model()

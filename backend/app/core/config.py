from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.config import get_config_value, resolve_path

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    APP_NAME: str = get_config_value("app", "name", default="RadiologyAI")
    APP_TITLE: str = get_config_value("app", "title", default="RadiologyAI API")
    APP_DESCRIPTION: str = get_config_value(
        "app",
        "description",
        default="Chest X-Ray Disease Classifier",
    )
    APP_VERSION: str = get_config_value("app", "version", default="1.0.0")
    DEBUG: bool = get_config_value("app", "debug", default=True)
    API_PREFIX: str = get_config_value("backend", "api_prefix", default="/api/v1")
    ALLOWED_ORIGINS: list[str] = get_config_value(
        "backend",
        "allowed_origins",
        default=["http://localhost:3000", "http://localhost:5173"],
    )
    MODEL_PATH: str = str(
        resolve_path(
            get_config_value(
                "paths",
                "model_path",
                default=str(ROOT / "ml/models/efficientnetb0_best.pth"),
            )
        )
    )
    FEEDBACK_FILE: str = str(
        resolve_path(
            get_config_value("paths", "feedback_file", default="data/feedback.json")
        )
    )
    MLFLOW_TRACKING_URI: str = get_config_value(
        "ops",
        "mlflow",
        "tracking_uri",
        default="sqlite:///mlflow.db",
    )
    IMAGE_SIZE: int = get_config_value("ml", "image_size", default=224)
    CLASS_NAMES: list[str] = get_config_value(
        "ml",
        "class_names",
        default=["Normal", "Pneumonia", "COVID19"],
    )
    RISK_MAP: dict[str, str] = get_config_value(
        "ml",
        "risk_map",
        default={"Normal": "Low", "Pneumonia": "High", "COVID19": "High"},
    )
    MODEL_NAME: str = get_config_value(
        "ml",
        "model",
        "architecture",
        default="EfficientNetB0",
    )
    MODEL_FRAMEWORK: str = get_config_value(
        "ml",
        "model",
        "framework",
        default="PyTorch 2.3.0",
    )
    ALLOWED_UPLOAD_TYPES: set[str] = set(
        get_config_value(
            "backend",
            "upload",
            "allowed_types",
            default=["image/jpeg", "image/png", "image/jpg"],
        )
    )
    MAX_UPLOAD_SIZE_MB: int = get_config_value(
        "backend",
        "upload",
        "max_size_mb",
        default=10,
    )
    FEEDBACK_ACCURACY_THRESHOLD: float = get_config_value(
        "backend",
        "feedback",
        "accuracy_threshold",
        default=0.8,
    )
    FEEDBACK_MIN_SAMPLES: int = get_config_value(
        "backend",
        "feedback",
        "min_feedback_samples",
        default=10,
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

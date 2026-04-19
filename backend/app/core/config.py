
import os
from pathlib import Path
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    APP_NAME: str = "RadiologyAI"
    DEBUG: bool = True
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    MODEL_PATH: str = str(ROOT / "ml/models/efficientnetb0_best.pth")
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    IMAGE_SIZE: int = 224
    CLASS_NAMES: list = ["Normal", "Pneumonia", "COVID19"]

    class Config:
        env_file = ".env"

settings = Settings()

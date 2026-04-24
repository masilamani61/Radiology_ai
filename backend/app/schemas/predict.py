from pydantic import BaseModel
from typing import List, Optional

class ClassProbability(BaseModel):
    class_name: str
    confidence: float

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    risk_level: str
    all_probabilities: List[ClassProbability]
    gradcam_base64: Optional[str] = None
    inference_time_ms: float
    mlflow_run_id: str = "local"
    disclaimer: str = "AI-assisted preliminary assessment only. Not a substitute for professional diagnosis."

class FeedbackRequest(BaseModel):
    prediction_id: str
    predicted_class: str
    correct_class: Optional[str] = None
    radiologist_confirmed: bool
    comments: Optional[str] = None

class FeedbackResponse(BaseModel):
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
    model_name: str
    framework: str
    image_size: int
    class_count: int

class ClassesResponse(BaseModel):
    classes: List[str]
    total: int

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from prometheus_client import Counter, Gauge
from backend.app.schemas.predict import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()

FEEDBACK_FILE = Path("data/feedback.json")
FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

# Prometheus metrics for feedback
feedback_total = Counter(
    "radiologyai_feedback_total",
    "Total feedback submissions",
    ["result"]  # correct / incorrect
)

model_accuracy_gauge = Gauge(
    "radiologyai_model_accuracy",
    "Model accuracy based on radiologist feedback (rolling)"
)

feedback_by_class = Counter(
    "radiologyai_feedback_by_class",
    "Feedback breakdown by predicted class",
    ["predicted_class", "result"]
)

correct_total   = 0
incorrect_total = 0


def update_accuracy_gauge():
    """Recompute accuracy from all feedback and update gauge."""
    global correct_total, incorrect_total
    total = correct_total + incorrect_total
    if total > 0:
        accuracy = correct_total / total
        model_accuracy_gauge.set(accuracy)
        logger.info(f"Model accuracy updated: {accuracy:.4f} ({correct_total}/{total})")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    POST /feedback
    Radiologist confirms or rejects AI prediction.
    Updates Prometheus accuracy gauge in real time.
    """
    global correct_total, incorrect_total

    try:
        is_correct = request.radiologist_confirmed

        # update counters
        if is_correct:
            correct_total += 1
            feedback_total.labels(result="correct").inc()
            feedback_by_class.labels(
                predicted_class=request.predicted_class,
                result="correct"
            ).inc()
        else:
            incorrect_total += 1
            feedback_total.labels(result="incorrect").inc()
            feedback_by_class.labels(
                predicted_class=request.predicted_class,
                result="incorrect"
            ).inc()

        update_accuracy_gauge()

        entry = {
            "timestamp"            : datetime.utcnow().isoformat(),
            "prediction_id"        : request.prediction_id,
            "predicted_class"      : request.predicted_class,
            "correct_class"        : request.correct_class,
            "radiologist_confirmed": request.radiologist_confirmed,
            "comments"             : request.comments,
            "is_correct"           : is_correct,
        }

        existing = []
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE) as f:
                existing = json.load(f)
        existing.append(entry)
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(existing, f, indent=2)

        total     = correct_total + incorrect_total
        accuracy  = correct_total / total if total > 0 else 0

        logger.info(
            f"Feedback: {request.predicted_class} "
            f"correct={is_correct} "
            f"accuracy={accuracy:.3f} ({correct_total}/{total})"
        )

        return FeedbackResponse(
            status  = "success",
            message = f"Feedback recorded. Model accuracy: {accuracy:.1%} ({total} samples)"
        )

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return FeedbackResponse(status="error", message=str(e))


@router.get("/feedback/stats")
async def get_feedback_stats():
    """GET /feedback/stats — returns accuracy stats from all feedback."""
    total     = correct_total + incorrect_total
    accuracy  = correct_total / total if total > 0 else 0

    # load from file for per-class breakdown
    class_stats = {}
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE) as f:
            entries = json.load(f)
        for cls in ["Normal", "Pneumonia", "COVID19"]:
            cls_entries = [e for e in entries if e["predicted_class"] == cls]
            cls_correct = sum(1 for e in cls_entries if e["is_correct"])
            class_stats[cls] = {
                "total"   : len(cls_entries),
                "correct" : cls_correct,
                "accuracy": round(cls_correct / len(cls_entries), 4) if cls_entries else None
            }

    return {
        "total_feedback"  : total,
        "correct"         : correct_total,
        "incorrect"       : incorrect_total,
        "overall_accuracy": round(accuracy, 4),
        "per_class"       : class_stats,
        "retrain_needed"  : accuracy < 0.80 and total >= 10
    }

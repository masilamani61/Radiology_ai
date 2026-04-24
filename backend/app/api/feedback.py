import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from prometheus_client import Counter, Gauge
from backend.app.core.config import settings
from backend.app.schemas.predict import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()

FEEDBACK_FILE = Path(settings.FEEDBACK_FILE)
FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

# Prometheus metrics
feedback_total = Counter(
    "radiologyai_feedback_total",
    "Total feedback submissions",
    ["result"]
)

model_accuracy_gauge = Gauge(
    "radiologyai_model_accuracy",
    "Model accuracy based on radiologist feedback"
)

feedback_by_class = Counter(
    "radiologyai_feedback_by_class",
    "Feedback by predicted class",
    ["predicted_class", "result"]
)

correct_total   = 0
incorrect_total = 0


def load_feedback_from_file():
    """
    Load existing feedback.json on startup and
    initialize all Prometheus metrics from historical data.
    Called once when the app starts.
    """
    global correct_total, incorrect_total

    if not FEEDBACK_FILE.exists():
        logger.info("No existing feedback.json found — starting fresh")
        model_accuracy_gauge.set(0)
        return

    try:
        with open(FEEDBACK_FILE) as f:
            entries = json.load(f)

        if not entries:
            model_accuracy_gauge.set(0)
            return

        # Rebuild counters from history
        for entry in entries:
            is_correct = entry.get("is_correct", entry.get("radiologist_confirmed", False))
            cls        = entry.get("predicted_class", "Unknown")

            if is_correct:
                correct_total += 1
                feedback_total.labels(result="correct").inc()
                feedback_by_class.labels(predicted_class=cls, result="correct").inc()
            else:
                incorrect_total += 1
                feedback_total.labels(result="incorrect").inc()
                feedback_by_class.labels(predicted_class=cls, result="incorrect").inc()

        # Set accuracy gauge
        total    = correct_total + incorrect_total
        accuracy = correct_total / total if total > 0 else 0
        model_accuracy_gauge.set(accuracy)

        logger.info(
            f"Feedback loaded from file: "
            f"{total} entries | accuracy={accuracy:.1%} "
            f"({correct_total} correct / {incorrect_total} incorrect)"
        )

    except Exception as e:
        logger.error(f"Failed to load feedback from file: {e}")
        model_accuracy_gauge.set(0)


def update_accuracy_gauge():
    """Recompute and update accuracy gauge."""
    global correct_total, incorrect_total
    total    = correct_total + incorrect_total
    accuracy = correct_total / total if total > 0 else 0
    model_accuracy_gauge.set(accuracy)
    logger.info(f"Accuracy updated: {accuracy:.4f} ({correct_total}/{total})")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    global correct_total, incorrect_total
    try:
        is_correct = request.radiologist_confirmed

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

        total    = correct_total + incorrect_total
        accuracy = correct_total / total if total > 0 else 0

        return FeedbackResponse(
            status  = "success",
            message = f"Feedback recorded. Accuracy: {accuracy:.1%} ({total} samples)"
        )

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return FeedbackResponse(status="error", message=str(e))
@router.get("/feedback/history")
async def get_feedback_history():
    """GET /feedback/history — returns all scan history for Patient History page."""
    if not FEEDBACK_FILE.exists():
        return {"history": [], "total": 0}
    try:
        with open(FEEDBACK_FILE) as f:
            entries = json.load(f)
        entries_sorted = sorted(entries, key=lambda x: x.get("timestamp",""), reverse=True)
        return {"history": entries_sorted, "total": len(entries_sorted)}
    except Exception as e:
        logger.error(f"History fetch failed: {e}")
        return {"history": [], "total": 0}

@router.get("/feedback/stats")
async def get_feedback_stats():
    total    = correct_total + incorrect_total
    accuracy = correct_total / total if total > 0 else 0

    class_stats = {}
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE) as f:
            entries = json.load(f)
        for cls in settings.CLASS_NAMES:
            cls_entries = [e for e in entries if e["predicted_class"] == cls]
            cls_correct = sum(1 for e in cls_entries if e.get("is_correct", False))
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
        "retrain_needed"  : (
            accuracy < settings.FEEDBACK_ACCURACY_THRESHOLD
            and total >= settings.FEEDBACK_MIN_SAMPLES
        ),
    }

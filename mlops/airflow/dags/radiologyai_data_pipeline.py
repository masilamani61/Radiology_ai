from datetime import date, datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param
import json
import os
import requests
import sys

DEFAULT_BASE = os.environ.get("RADIOLOGYAI_BASE", "/opt/radiologyai")
if DEFAULT_BASE not in sys.path:
    sys.path.insert(0, DEFAULT_BASE)

from shared.config import get_config_value

# ── Detect base path from environment or fallback ────────────

BASE   = os.environ.get(
    "RADIOLOGYAI_BASE",
    get_config_value("ops", "airflow", "base_dir", default=DEFAULT_BASE),
)
VENV   = f"{BASE}/venv/bin/activate"
PREFIX = f"cd {BASE} && source {VENV} && export PYTHONPATH={BASE}"


def as_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported start_date value: {value!r}")

default_args = {
    "owner"          : "radiologyai",
    "depends_on_past": False,
    "start_date"     : as_datetime(
        get_config_value("ops", "airflow", "start_date", default="2025-01-01")
    ),
    "retries"        : get_config_value("ops", "airflow", "retries", default=1),
    "retry_delay"    : timedelta(
        minutes=get_config_value(
            "ops",
            "airflow",
            "retry_delay_minutes",
            default=5,
        )
    ),
}

dag = DAG(
    dag_id            = "radiologyai_pipeline",
    default_args      = default_args,
    description       = "RadiologyAI feedback-based retraining pipeline",
    schedule_interval = get_config_value(
        "ops",
        "airflow",
        "schedule_interval",
        default="@daily",
    ),
    catchup           = False,
    tags              = ["radiologyai", "mlops"],
    params            = {
        "batch_size"          : Param(get_config_value("ml", "train", "batch_size", default=32), type="integer", title="Batch Size",           minimum=8,      maximum=128),
        "epochs_phase1"       : Param(get_config_value("ml", "train", "epochs_phase1", default=5), type="integer", title="Phase 1 Epochs",       minimum=1,      maximum=20),
        "epochs_phase2"       : Param(get_config_value("ml", "train", "epochs_phase2", default=15), type="integer", title="Phase 2 Epochs",       minimum=1,      maximum=50),
        "lr_phase1"           : Param(get_config_value("ml", "train", "lr_phase1", default=0.001), type="number",  title="LR Phase 1",           minimum=0.0001, maximum=0.01),
        "lr_phase2"           : Param(get_config_value("ml", "train", "lr_phase2", default=0.0001), type="number",  title="LR Phase 2",           minimum=0.00001,maximum=0.001),
        "accuracy_threshold"  : Param(get_config_value("backend", "feedback", "accuracy_threshold", default=0.80), type="number",  title="Accuracy Threshold",   minimum=0.5,    maximum=0.99),
        "min_feedback_samples": Param(get_config_value("backend", "feedback", "min_feedback_samples", default=10), type="integer", title="Min Feedback Samples", minimum=5,      maximum=100),
        "force_retrain"       : Param(False, type="boolean", title="Force Retrain"),
    },
)

validate = BashOperator(
    task_id      = "validate_data",
    bash_command = f"{PREFIX} && python ml/src/data/validate.py",
    dag          = dag,
)

def check_feedback_accuracy(**kwargs):
    params      = kwargs["params"]
    threshold   = params["accuracy_threshold"]
    min_samples = params["min_feedback_samples"]
    force       = params["force_retrain"]

    if force:
        print("Force retrain enabled")
        kwargs["ti"].xcom_push(key="retrain_needed", value=True)
        kwargs["ti"].xcom_push(key="accuracy",       value=None)
        kwargs["ti"].xcom_push(key="total_feedback", value=0)
        return

    try:
        res      = requests.get(
            get_config_value(
                "ops",
                "airflow",
                "feedback_stats_url",
                default="http://host.docker.internal:8005/api/v1/feedback/stats",
            ),
            timeout=5
        )
        stats    = res.json()
        accuracy = stats.get("overall_accuracy", 1.0)
        total    = stats.get("total_feedback",   0)
        retrain  = accuracy < threshold and total >= min_samples
        print(f"Accuracy: {accuracy:.1%} | Total: {total} | Retrain: {retrain}")
        kwargs["ti"].xcom_push(key="retrain_needed", value=retrain)
        kwargs["ti"].xcom_push(key="accuracy",       value=accuracy)
        kwargs["ti"].xcom_push(key="total_feedback", value=total)
    except Exception as e:
        print(f"API unreachable: {e} — skipping retrain")
        kwargs["ti"].xcom_push(key="retrain_needed", value=False)
        kwargs["ti"].xcom_push(key="accuracy",       value=None)
        kwargs["ti"].xcom_push(key="total_feedback", value=0)

feedback_check = PythonOperator(
    task_id         = "check_feedback_accuracy",
    python_callable = check_feedback_accuracy,
    dag             = dag,
)

def check_drift(**kwargs):
    baseline = Path(BASE) / "data/processed/baseline_stats.json"
    if not baseline.exists():
        print("No baseline found — no drift check")
        kwargs["ti"].xcom_push(key="drift_detected", value=False)
        return
    with open(baseline) as f:
        data = json.load(f)
    print(f"Baseline classes: {list(data.keys())}")
    kwargs["ti"].xcom_push(key="drift_detected", value=False)

drift_check = PythonOperator(
    task_id         = "check_data_drift",
    python_callable = check_drift,
    dag             = dag,
)

def branch_retrain(**kwargs):
    retrain = kwargs["ti"].xcom_pull(task_ids="check_feedback_accuracy", key="retrain_needed")
    drift   = kwargs["ti"].xcom_pull(task_ids="check_data_drift",        key="drift_detected")
    print(f"Retrain needed: {retrain} | Drift: {drift}")
    return "retrain_model" if (retrain or drift) else "skip_retrain"

branch = BranchPythonOperator(
    task_id         = "branch_retrain_or_skip",
    python_callable = branch_retrain,
    dag             = dag,
)

retrain = BashOperator(
    task_id      = "retrain_model",
    bash_command = (
        f"{PREFIX} && python ml/src/models/train.py"
        " --batch_size {{ params.batch_size }}"
        " --epochs_phase1 {{ params.epochs_phase1 }}"
        " --epochs_phase2 {{ params.epochs_phase2 }}"
        " --lr_phase1 {{ params.lr_phase1 }}"
        " --lr_phase2 {{ params.lr_phase2 }}"
        f" --mlflow_uri {get_config_value('ops', 'airflow', 'mlflow_service_url', default='http://host.docker.internal:5005')}"
    ),
    dag = dag,
)

skip = EmptyOperator(task_id="skip_retrain", dag=dag)

evaluate = BashOperator(
    task_id      = "evaluate_model",
    bash_command = (
        f"{PREFIX} && python ml/src/evaluation/evaluation.py"
        f" --mlflow_uri {get_config_value('ops', 'airflow', 'mlflow_service_url', default='http://host.docker.internal:5005')}"
    ),
    dag = dag,
)

export = BashOperator(
    task_id      = "export_model",
    bash_command = f"{PREFIX} && python ml/src/models/exports.py",
    dag          = dag,
)

def notify_completion(**kwargs):
    accuracy = kwargs["ti"].xcom_pull(
        task_ids="check_feedback_accuracy", key="accuracy"
    )
    print(f"Pipeline complete! Accuracy was: {accuracy}")
    print(f"New model exported from: {BASE}/ml/models/")

notify = PythonOperator(
    task_id         = "notify_completion",
    python_callable = notify_completion,
    trigger_rule    = "none_failed_min_one_success",
    dag             = dag,
)

validate >> [feedback_check, drift_check] >> branch
branch >> retrain >> evaluate >> export >> notify
branch >> skip >> notify

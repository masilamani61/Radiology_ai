from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param
import json
import requests
from pathlib import Path

default_args = {
    "owner"          : "radiologyai",
    "depends_on_past": False,
    "start_date"     : datetime(2025, 1, 1),
    "retries"        : 1,
    "retry_delay"    : timedelta(minutes=5),
}

BASE   = "/data/data/DA25S005/Radiology_ai"
PREFIX = (
    f"cd {BASE} && source venv/bin/activate && "
    f"export PYTHONPATH={BASE} && "
    f"export CUDA_VISIBLE_DEVICES=0"
)

dag = DAG(
    dag_id            = "radiologyai_pipeline",
    default_args      = default_args,
    description       = "RadiologyAI — feedback-based retraining with configurable params",
    schedule = "@daily",
    catchup           = False,
    tags              = ["radiologyai", "mlops", "feedback"],
    params            = {
        "batch_size": Param(
            128,
            type       = "integer",
            title      = "Batch Size",
            description= "Training batch size (32/64/128/256)",
            minimum    = 16,
            maximum    = 256,
        ),
        "epochs_phase1": Param(
            5,
            type       = "integer",
            title      = "Phase 1 Epochs (frozen backbone)",
            description= "Epochs to train classifier head only",
            minimum    = 1,
            maximum    = 20,
        ),
        "epochs_phase2": Param(
            15,
            type       = "integer",
            title      = "Phase 2 Epochs (full fine-tune)",
            description= "Epochs for full model fine-tuning",
            minimum    = 1,
            maximum    = 50,
        ),
        "lr_phase1": Param(
            0.001,
            type       = "number",
            title      = "Learning Rate Phase 1",
            description= "LR for classifier head training",
            minimum    = 0.0001,
            maximum    = 0.01,
        ),
        "lr_phase2": Param(
            0.0001,
            type       = "number",
            title      = "Learning Rate Phase 2",
            description= "LR for full fine-tuning",
            minimum    = 0.00001,
            maximum    = 0.001,
        ),
        "accuracy_threshold": Param(
            0.80,
            type       = "number",
            title      = "Accuracy Threshold for Retraining",
            description= "Retrain if radiologist feedback accuracy drops below this",
            minimum    = 0.50,
            maximum    = 0.99,
        ),
        "min_feedback_samples": Param(
            10,
            type       = "integer",
            title      = "Min Feedback Samples to Trigger Retrain",
            description= "Minimum radiologist feedback count before retraining",
            minimum    = 5,
            maximum    = 100,
        ),
        "force_retrain": Param(
            False,
            type       = "boolean",
            title      = "Force Retrain",
            description= "Force retraining regardless of accuracy/drift",
        ),
    },
)

# ── Task 1: Validate ──────────────────────────────────────────
validate = BashOperator(
    task_id      = "validate_data",
    bash_command = f"{PREFIX} && python ml/src/data/validate.py",
    dag          = dag,
)

# ── Task 2: Check feedback accuracy ──────────────────────────
def check_feedback_accuracy(**kwargs):
    params    = kwargs["params"]
    threshold = params["accuracy_threshold"]
    min_samples = params["min_feedback_samples"]
    force     = params["force_retrain"]

    print(f"Parameters received:")
    print(f"  accuracy_threshold    : {threshold}")
    print(f"  min_feedback_samples  : {min_samples}")
    print(f"  force_retrain         : {force}")

    if force:
        print("Force retrain enabled — skipping accuracy check")
        kwargs["ti"].xcom_push(key="retrain_needed", value=True)
        kwargs["ti"].xcom_push(key="accuracy",       value=None)
        kwargs["ti"].xcom_push(key="total_feedback", value=0)
        return

    try:
        res      = requests.get(
            "http://localhost:8005/api/v1/feedback/stats",
            timeout=5
        )
        stats    = res.json()
        accuracy = stats.get("overall_accuracy", 1.0)
        total    = stats.get("total_feedback",   0)
        retrain  = accuracy < threshold and total >= min_samples

        print(f"Feedback stats:")
        print(f"  Total    : {total}")
        print(f"  Accuracy : {accuracy:.1%}")
        print(f"  Threshold: {threshold:.1%}")
        print(f"  Retrain? : {retrain}")

        kwargs["ti"].xcom_push(key="retrain_needed", value=retrain)
        kwargs["ti"].xcom_push(key="accuracy",       value=accuracy)
        kwargs["ti"].xcom_push(key="total_feedback", value=total)

    except Exception as e:
        print(f"Could not reach API: {e} — skipping retrain")
        kwargs["ti"].xcom_push(key="retrain_needed", value=False)
        kwargs["ti"].xcom_push(key="accuracy",       value=None)
        kwargs["ti"].xcom_push(key="total_feedback", value=0)

feedback_check = PythonOperator(
    task_id         = "check_feedback_accuracy",
    python_callable = check_feedback_accuracy,
    dag             = dag,
)

# ── Task 3: Check drift ───────────────────────────────────────
def check_drift(**kwargs):
    baseline_path = Path(f"{BASE}/data/processed/baseline_stats.json")
    if not baseline_path.exists():
        kwargs["ti"].xcom_push(key="drift_detected", value=False)
        return
    with open(baseline_path) as f:
        baseline = json.load(f)
    print(f"Baseline loaded: {list(baseline.keys())}")
    kwargs["ti"].xcom_push(key="drift_detected", value=False)

drift_check = PythonOperator(
    task_id         = "check_data_drift",
    python_callable = check_drift,
    dag             = dag,
)

# ── Task 4: Branch ────────────────────────────────────────────
def branch_retrain(**kwargs):
    retrain = kwargs["ti"].xcom_pull(
        task_ids="check_feedback_accuracy", key="retrain_needed"
    )
    drift = kwargs["ti"].xcom_pull(
        task_ids="check_data_drift", key="drift_detected"
    )
    accuracy = kwargs["ti"].xcom_pull(
        task_ids="check_feedback_accuracy", key="accuracy"
    )
    total = kwargs["ti"].xcom_pull(
        task_ids="check_feedback_accuracy", key="total_feedback"
    )

    print(f"Branch decision:")
    print(f"  retrain_needed : {retrain}")
    print(f"  drift_detected : {drift}")
    print(f"  accuracy       : {accuracy}")
    print(f"  total_feedback : {total}")

    if retrain or drift:
        print("DECISION → RETRAIN")
        return "retrain_model"
    print("DECISION → SKIP")
    return "skip_retrain"

branch = BranchPythonOperator(
    task_id         = "branch_retrain_or_skip",
    python_callable = branch_retrain,
    dag             = dag,
)

# ── Task 5a: Retrain with params ──────────────────────────────
def get_retrain_command(**kwargs):
    params = kwargs["params"]
    cmd = (
        f"{PREFIX} && python ml/src/models/train.py"
        f" --batch_size {params['batch_size']}"
        f" --epochs_phase1 {params['epochs_phase1']}"
        f" --epochs_phase2 {params['epochs_phase2']}"
        f" --lr_phase1 {params['lr_phase1']}"
        f" --lr_phase2 {params['lr_phase2']}"
        f" --mlflow_uri http://localhost:5005"
    )
    print(f"Running: {cmd}")
    return cmd

retrain = BashOperator(
    task_id      = "retrain_model",
    bash_command = (
        f"{PREFIX} && python ml/src/models/train.py"
        " --batch_size {{ params.batch_size }}"
        " --epochs_phase1 {{ params.epochs_phase1 }}"
        " --epochs_phase2 {{ params.epochs_phase2 }}"
        " --lr_phase1 {{ params.lr_phase1 }}"
        " --lr_phase2 {{ params.lr_phase2 }}"
        " --mlflow_uri http://localhost:5005"
    ),
    dag = dag,
)

# ── Task 5b: Skip ─────────────────────────────────────────────
skip = EmptyOperator(task_id="skip_retrain", dag=dag)

# ── Task 6: Evaluate ──────────────────────────────────────────
evaluate = BashOperator(
    task_id      = "evaluate_model",
    bash_command = f"{PREFIX} && python ml/src/evaluation/evaluate.py --mlflow_uri http://localhost:5005",
    dag          = dag,
)

# ── Task 7: Export ────────────────────────────────────────────
export = BashOperator(
    task_id      = "export_model",
    bash_command = f"{PREFIX} && python ml/src/models/export.py",
    dag          = dag,
)

# ── Task 8: Notify ────────────────────────────────────────────
def notify_completion(**kwargs):
    accuracy = kwargs["ti"].xcom_pull(
        task_ids="check_feedback_accuracy", key="accuracy"
    )
    params = kwargs["params"]
    print("Pipeline complete!")
    print(f"  Previous accuracy : {accuracy}")
    print(f"  Batch size used   : {params['batch_size']}")
    print(f"  Phase1 epochs     : {params['epochs_phase1']}")
    print(f"  Phase2 epochs     : {params['epochs_phase2']}")
    print("New model exported and registered in MLflow.")

notify = PythonOperator(
    task_id         = "notify_completion",
    python_callable = notify_completion,
    trigger_rule    = "none_failed_min_one_success",
    dag             = dag,
)

# ── Wiring ────────────────────────────────────────────────────
validate >> [feedback_check, drift_check] >> branch
branch >> retrain >> evaluate >> export >> notify
branch >> skip >> notify

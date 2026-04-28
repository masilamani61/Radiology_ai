# RadiologyAI
### Name : V G Masilamani
### Roll No : DA25S005
---



RadiologyAI is an end-to-end chest X-ray classification and MLOps project. It combines a React frontend, FastAPI backend, PyTorch model pipeline, MLflow experiment tracking, DVC reproducibility, Airflow retraining orchestration, and Prometheus/Grafana monitoring.

The application is designed to classify chest X-ray images into:

- Normal
- Pneumonia
- COVID19

In addition to inference, the project includes model evaluation, explainability with Grad-CAM, feedback collection, drift reporting, PDF report generation, and operational monitoring.

---

## Features

### Application Features

- Chest X-ray upload and analysis
- Prediction with confidence score
- Risk-level classification
- Grad-CAM explainability heatmap
- Result page with class probability breakdown
- Radiologist feedback submission
- Clinical PDF report generation
- Feedback history view
- Drift summary and drift report endpoints
- MLflow run comparison support

### MLOps Features

- Config-driven project setup using `config.yaml`
- DVC pipeline for data and model workflow
- MLflow tracking and artifact logging
- Airflow DAG for retraining orchestration
- Prometheus metrics collection
- Grafana dashboard provisioning
- Node exporter and Nginx exporter integration
- Docker Compose deployment for the full stack

---

## Tech Stack

### Frontend

- React
- Axios
- Recharts
- React Dropzone
- React Hot Toast
- Nginx

### Backend

- FastAPI
- Pydantic Settings
- Pillow
- NumPy
- Prometheus FastAPI Instrumentator

### Machine Learning

- PyTorch
- scikit-learn
- MLflow
- DVC
- Matplotlib

### MLOps and Observability

- Docker Compose
- Apache Airflow
- Prometheus
- Grafana
- Alertmanager
- node_exporter
- nginx-prometheus-exporter

---

## Project Structure

```text
Radiology_ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   └── services/
│   └── requirements.txt
├── frontend/
│   ├── public/
│   └── src/
├── ml/
│   ├── experiments/
│   ├── models/
│   └── src/
├── mlops/
│   ├── airflow/
│   ├── grafana/
│   └── prometheus/
├── docker/
├── data/
│   ├── raw/
│   └── processed/
├── logs/
├── scripts/
├── shared/
├── tests/
├── config.yaml
├── docker-compose.yml
├── dvc.yaml
├── MLproject
├── start.sh
└── stop.sh
```

---

## Core Modules

### Frontend

The frontend is a React single-page application with the following main pages:

- `UploadPage` for X-ray upload
- `ResultPage` for prediction results and feedback
- `MonitoringPage` for system health links
- `HistoryPage` for feedback history
- `DriftPage` for drift-related outputs
- `ModelComparePage` for MLflow run comparison

Main entry:
- `frontend/src/App.js`

### Backend

The backend is a FastAPI application that:

- loads the trained model on startup
- serves prediction APIs
- records radiologist feedback
- exposes health endpoints
- exposes Prometheus metrics
- generates PDF reports
- returns drift and MLflow comparison data

Main entry:
- `backend/app/main.py`

### ML Pipeline

The ML code includes:

- model training
- evaluation
- explainability generation
- export to TorchScript
- MLflow registration support

Key files:
- `ml/src/models/train.py`
- `ml/src/evaluation/evaluation.py`
- `ml/src/explainability/gradcam.py`
- `ml/src/models/exports.py`

### Airflow

Airflow is used for feedback-aware retraining orchestration.

DAG:
- `mlops/airflow/dags/radiologyai_data_pipeline.py`

The DAG performs:

- data validation
- feedback accuracy check
- drift check
- retrain/skip branching
- evaluation
- export
- completion notification

---

## Configuration

The project uses a shared configuration system based on:

- `config.yaml`
- `shared/config.py`

This central configuration controls:

- app metadata
- backend host and port
- frontend URLs
- model paths
- ML hyperparameters
- evaluation thresholds
- Airflow schedule and retry settings
- MLflow, Prometheus, Grafana, Alertmanager, and node exporter ports

---

## Supported Classes

Configured in `config.yaml`:

- Normal
- Pneumonia
- COVID19

Risk mapping:

- Normal → Low
- Pneumonia → High
- COVID19 → High

---

## Main API Endpoints

### Health

- `GET /health`
- `GET /ready`
- `GET /classes`

### Prediction

- `POST /api/v1/predict`

Accepts:
- `image/jpeg`
- `image/png`
- `image/jpg`

Default max upload size:
- 10 MB

### Feedback

- `POST /api/v1/feedback`
- `GET /api/v1/feedback/history`
- `GET /api/v1/feedback/stats`

### Reports

- `GET /api/v1/drift/summary`
- `GET /api/v1/drift/report/{cls_name}`
- `GET /api/v1/mlflow/runs`
- `POST /api/v1/report/pdf`

### Metrics

- `GET /metrics`

---

## DVC Pipeline

Defined in `dvc.yaml`.

Stages include:

- `download`
- `validate`
- `preprocess`
- `train`
- `evaluate`
- `explain`
- `export`
- `drift_detection`
- `data_validation`

Typical DVC commands:

```bash
dvc repro
dvc dag
dvc status
```

---

## MLflow

MLflow is used for:

- experiment tracking
- metric logging
- artifact logging
- model comparison
- model registration support

Default service URL from config:
- `http://localhost:5005`

---

## Monitoring Stack

The monitoring stack includes:

- Prometheus
- Grafana
- Alertmanager
- node_exporter
- nginx-prometheus-exporter

Prometheus scrapes:

- backend
- frontend nginx exporter
- grafana
- prometheus
- node exporter
- alertmanager

Grafana is provisioned from the files under:

- `mlops/grafana/provisioning/`
- `mlops/grafana/dashboards/`

---

## Docker Compose Services

Defined in `docker-compose.yml`.

Services:

- `frontend`
- `backend`
- `mlflow`
- `prometheus`
- `grafana`
- `nginx_exporter`
- `alertmanager`
- `airflow`
- `node_exporter`

Default ports from config:

- Frontend: `3000`
- Backend: `8005`
- MLflow: `5005`
- Grafana: `3001`
- Prometheus: `9090`
- Airflow: `8080`
- Alertmanager: `9093`
- node_exporter: `9100`

---

## How to Run

### Option 1: Docker Compose

```bash
docker compose up --build
```

To run in detached mode:

```bash
docker compose up --build -d
```

To stop:

```bash
docker compose down
```

### Option 2: Local Startup Script

```bash
bash start.sh
```

To stop local processes:

```bash
bash stop.sh
```

---

## Backend Startup Flow

On startup the FastAPI app:

1. loads configuration
2. initializes logging
3. registers Prometheus instrumentation
4. loads the model through `ModelService`
5. loads historical feedback
6. registers prediction, feedback, health, and report routers

---

## Prediction Workflow

1. User uploads a chest X-ray image in the frontend
2. Frontend calls `POST /api/v1/predict`
3. Backend validates file type and size
4. Backend loads and preprocesses the image
5. PyTorch model performs inference
6. Grad-CAM heatmap is generated
7. Response is returned with:
   - predicted class
   - confidence
   - risk level
   - class probabilities
   - Grad-CAM image
   - inference latency

---

## Feedback Workflow

1. User confirms or rejects the prediction
2. Frontend submits feedback to `POST /api/v1/feedback`
3. Backend stores the feedback in `data/feedback.json`
4. Feedback metrics are updated
5. Airflow can use feedback stats to determine retraining need

---

## PDF Report Generation

The backend supports PDF clinical report generation through:

- `POST /api/v1/report/pdf`

The report includes:

- patient information
- AI diagnosis
- class probabilities
- Grad-CAM image
- inference details
- report identifier
- disclaimer

---

## Testing

Backend test file:
- `tests/test_api.py`

Frontend test file:
- `frontend/src/App.test.js`

To run Python tests if environment is ready:

```bash
pytest
```

To run frontend tests:

```bash
cd frontend
npm test
```

---

## Important Files

### Config and Shared Utilities

- `config.yaml`
- `shared/config.py`
- `scripts/sync_config.py`

### Backend

- `backend/app/main.py`
- `backend/app/api/predict.py`
- `backend/app/api/feedback.py`
- `backend/app/api/health.py`
- `backend/app/api/reports.py`
- `backend/app/services/model_service.py`
- `backend/app/core/config.py`
- `backend/app/core/metrics.py`

### Frontend

- `frontend/src/App.js`
- `frontend/src/services/api.js`
- `frontend/src/pages/UploadPage.js`
- `frontend/src/pages/ResultPage.js`
- `frontend/src/pages/MonitoringPage.js`

### ML

- `ml/src/models/train.py`
- `ml/src/models/model.py`
- `ml/src/models/exports.py`
- `ml/src/evaluation/evaluation.py`
- `ml/src/explainability/gradcam.py`

### MLOps

- `mlops/airflow/dags/radiologyai_data_pipeline.py`
- `mlops/prometheus/prometheus.yml`
- `mlops/grafana/dashboards/`
- `docker-compose.yml`

---

## Current Repository Notes

This repository currently includes not only source code but also project deliverables and supporting artifacts such as:

- `Radiologyai_hld.pdf`
- `LLD___API_endpoints.pdf`
- `AI_disclosure_final.pdf`
- `RADIOLOGY_Report.pdf`
- `MLOPS_FINAL.mov`

These appear to be supporting documentation and presentation assets for the project.

---

## Limitations

Current limitations visible from the repository include:

- feedback is stored in a JSON file rather than a database
- authentication and authorization are not implemented
- some drift and report endpoints depend on generated artifacts existing
- observability panels may remain idle until traffic is generated
- several runtime paths assume local or container-based execution

---

## Future Improvements

Possible next improvements:

- move feedback storage to PostgreSQL
- add user authentication and role-based authorization
- improve frontend UX and responsiveness
- add more model classes
- strengthen error handling and validation
- add model version rollback and better registry controls
- improve production deployment hardening
- add richer alerts and SLA-oriented monitoring

---

## Summary

RadiologyAI is more than a simple inference demo. It is a full-stack AI + MLOps project that includes:

- model training and evaluation
- inference serving
- explainability
- feedback collection
- retraining orchestration
- experiment tracking
- monitoring and observability
- deployment support

It is a strong end-to-end project for demonstration, academic evaluation, and future extension into a more production-ready medical AI platform.

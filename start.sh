#!/bin/bash
set -a
python3 scripts/sync_config.py
source .env
set +a

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting RadiologyAI..."

source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

# Kill existing processes
pkill -f "uvicorn" 2>/dev/null
pkill -f "mlflow" 2>/dev/null
pkill -f "prometheus" 2>/dev/null
pkill -f "grafana" 2>/dev/null
pkill -f "airflow" 2>/dev/null
sleep 2

# 1. MLflow
echo "Starting MLflow on :${MLFLOW_PORT}..."
mlflow server --host "${MLFLOW_HOST}" --port "${MLFLOW_PORT}" \
  --backend-store-uri "${MLFLOW_TRACKING_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
  > logs/mlflow.log 2>&1 &
sleep 3

# 2. Backend
echo "Starting FastAPI backend on :${BACKEND_PORT}..."
uvicorn backend.app.main:app \
  --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" \
  > logs/backend.log 2>&1 &
sleep 3

# 3. Prometheus
echo "Starting Prometheus on :${PROMETHEUS_PORT}..."
./prometheus/prometheus \
  --config.file=mlops/prometheus/prometheus.yml \
  --storage.tsdb.path=./prometheus_bin/data \
  --web.listen-address="0.0.0.0:${PROMETHEUS_PORT}" \
  > logs/prometheus.log 2>&1 &
sleep 2

# 4. Grafana
echo "Starting Grafana on :${GRAFANA_PORT}..."
./grafana/bin/grafana server \
  --homepath ./grafana \
  --configOverrides "cfg:server.http_port=${GRAFANA_PORT}" \
  > logs/grafana.log 2>&1 &
sleep 2

# 5. Airflow
echo "Starting Airflow on :${AIRFLOW_PORT}..."
export AIRFLOW_HOME="${PROJECT_ROOT}/mlops/airflow"
airflow standalone \
  > logs/airflow.log 2>&1 &
sleep 2

# 6. Frontend
echo "Starting React frontend on :${FRONTEND_PORT}..."
cd frontend && npm start \
  > ../logs/frontend.log 2>&1 &
cd ..

echo ""
echo "All services started!"
echo "  Frontend   : http://localhost:${FRONTEND_PORT}"
echo "  Backend    : http://localhost:${BACKEND_PORT}"
echo "  MLflow     : http://localhost:${MLFLOW_PORT}"
echo "  Prometheus : http://localhost:${PROMETHEUS_PORT}"
echo "  Grafana    : http://localhost:${GRAFANA_PORT}"
echo "  Airflow    : http://localhost:${AIRFLOW_PORT}"
echo ""
echo "Checking health..."
sleep 5
curl -s "http://localhost:${BACKEND_PORT}/health" && echo " Backend OK"
curl -s "http://localhost:${PROMETHEUS_PORT}/-/healthy" && echo " Prometheus OK"
curl -s "http://localhost:${GRAFANA_PORT}/api/health" | python3 -m json.tool | grep version && echo " Grafana OK"
curl -s "http://localhost:${MLFLOW_PORT}/health" && echo " MLflow OK"

# Alertmanager
echo "Starting Alertmanager on :${ALERTMANAGER_PORT}..."
./alertmanager_bin/alertmanager \
  --config.file=mlops/prometheus/alertmanager.yml \
  --storage.path=./alertmanager_bin/data \
  --web.listen-address="0.0.0.0:${ALERTMANAGER_PORT}" \
  
sleep 2
echo "  Alertmanager : http://localhost:${ALERTMANAGER_PORT}"

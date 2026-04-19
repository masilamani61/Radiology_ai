#!/bin/bash
echo "Starting RadiologyAI..."
cd /data/data/DA25S005/Radiology_ai
source venv/bin/activate
export PYTHONPATH=/data/data/DA25S005/Radiology_ai

# Kill existing processes
pkill -f "uvicorn" 2>/dev/null
pkill -f "mlflow" 2>/dev/null
pkill -f "prometheus" 2>/dev/null
pkill -f "grafana" 2>/dev/null
pkill -f "airflow" 2>/dev/null
sleep 2

# 1. MLflow
echo "Starting MLflow on :5005..."
mlflow server --host 0.0.0.0 --port 5005 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  > logs/mlflow.log 2>&1 &
sleep 3

# 2. Backend
echo "Starting FastAPI backend on :8005..."
uvicorn backend.app.main:app \
  --host 0.0.0.0 --port 8005 \
  > logs/backend.log 2>&1 &
sleep 3

# 3. Prometheus
echo "Starting Prometheus on :9090..."
./prometheus/prometheus \
  --config.file=mlops/prometheus/prometheus.yml \
  --storage.tsdb.path=./prometheus_bin/data \
  --web.listen-address=0.0.0.0:9090 \
  > logs/prometheus.log 2>&1 &
sleep 2

# 4. Grafana
echo "Starting Grafana on :3001..."
./grafana/bin/grafana server \
  --homepath ./grafana \
  --configOverrides cfg:server.http_port=3001 \
  > logs/grafana.log 2>&1 &
sleep 2

# 5. Airflow
echo "Starting Airflow on :8080..."
export AIRFLOW_HOME=/data/data/DA25S005/Radiology_ai/mlops/airflow
airflow standalone \
  > logs/airflow.log 2>&1 &
sleep 2

# 6. Frontend
echo "Starting React frontend on :3000..."
cd frontend && npm start \
  > ../logs/frontend.log 2>&1 &
cd ..

echo ""
echo "All services started!"
echo "  Frontend   : http://localhost:3000"
echo "  Backend    : http://localhost:8005"
echo "  MLflow     : http://localhost:5005"
echo "  Prometheus : http://localhost:9090"
echo "  Grafana    : http://localhost:3001"
echo "  Airflow    : http://localhost:8080"
echo ""
echo "Checking health..."
sleep 5
curl -s http://localhost:8005/health && echo " Backend OK"
curl -s http://localhost:9090/-/healthy && echo " Prometheus OK"
curl -s http://localhost:3001/api/health | python3 -m json.tool | grep version && echo " Grafana OK"
curl -s http://localhost:5005/health && echo " MLflow OK"

# Alertmanager
echo "Starting Alertmanager on :9093..."
./alertmanager_bin/alertmanager \
  --config.file=mlops/prometheus/alertmanager.yml \
  --storage.path=./alertmanager_bin/data \
  --web.listen-address=0.0.0.0:9093 \
  
sleep 2
echo "  Alertmanager : http://localhost:9093"

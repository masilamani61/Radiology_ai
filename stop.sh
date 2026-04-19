#!/bin/bash
echo "Stopping all RadiologyAI services..."
pkill -f "uvicorn" 2>/dev/null && echo "Backend stopped"
pkill -f "mlflow" 2>/dev/null && echo "MLflow stopped"
pkill -f "prometheus" 2>/dev/null && echo "Prometheus stopped"
pkill -f "grafana" 2>/dev/null && echo "Grafana stopped"
pkill -f "airflow" 2>/dev/null && echo "Airflow stopped"
pkill -f "react-scripts" 2>/dev/null && echo "Frontend stopped"
echo "All stopped."

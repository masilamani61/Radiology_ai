FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow

RUN mkdir -p /mlflow

CMD ["mlflow", "server",  "--host", "0.0.0.0",   "--port", "5005",  "--backend-store-uri", "sqlite:////mlflow/mlflow.db",   "--default-artifact-root", "/mlflow/mlruns",     "--serve-artifacts"]
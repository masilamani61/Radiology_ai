from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import ROOT, get_config_value


def write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    backend_port = str(get_config_value("backend", "port"))
    frontend_port = str(get_config_value("frontend", "port"))
    mlflow_port = str(get_config_value("ops", "mlflow", "port"))
    grafana_port = str(get_config_value("ops", "grafana", "port"))
    prometheus_port = str(get_config_value("ops", "prometheus", "port"))
    airflow_port = str(get_config_value("ops", "airflow", "port"))
    alertmanager_port = str(get_config_value("ops", "alertmanager", "port"))
    node_exporter_port = str(get_config_value("ops", "node_exporter", "port"))

    root_env = {
        "BACKEND_PORT": backend_port,
        "BACKEND_HOST": str(get_config_value("backend", "host")),
        "BACKEND_API_PREFIX": str(get_config_value("backend", "api_prefix")),
        "FRONTEND_PORT": frontend_port,
        "FRONTEND_API_URL": str(get_config_value("frontend", "api_url")),
        "MLFLOW_PORT": mlflow_port,
        "MLFLOW_HOST": str(get_config_value("ops", "mlflow", "host")),
        "MLFLOW_TRACKING_URI": str(get_config_value("ops", "mlflow", "tracking_uri")),
        "MLFLOW_SERVICE_URL": str(get_config_value("ops", "mlflow", "service_url")),
        "MLFLOW_ARTIFACT_ROOT": str(get_config_value("ops", "mlflow", "artifact_root")),
        "PROMETHEUS_PORT": prometheus_port,
        "GRAFANA_PORT": grafana_port,
        "GRAFANA_ADMIN_USER": str(get_config_value("ops", "grafana", "admin_user")),
        "GRAFANA_ADMIN_PASSWORD": str(get_config_value("ops", "grafana", "admin_password")),
        "AIRFLOW_PORT": airflow_port,
        "AIRFLOW_BASE_DIR": str(get_config_value("ops", "airflow", "base_dir")),
        "ALERTMANAGER_PORT": alertmanager_port,
        "NODE_EXPORTER_PORT": node_exporter_port,
    }
    frontend_env = {
        "REACT_APP_API_URL": str(get_config_value("frontend", "api_url")),
        "REACT_APP_API_PREFIX": str(get_config_value("backend", "api_prefix")),
        "REACT_APP_GRAFANA_URL": str(get_config_value("frontend", "grafana_url")),
        "REACT_APP_MLFLOW_URL": str(get_config_value("frontend", "mlflow_url")),
        "PORT": frontend_port,
    }

    write_env_file(ROOT / ".env", root_env)
    write_env_file(ROOT / "frontend" / ".env", frontend_env)


if __name__ == "__main__":
    main()

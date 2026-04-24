from prometheus_client import Counter, Gauge, Histogram


prediction_total = Counter(
    "radiologyai_predictions_total",
    "Total number of completed predictions",
    ["predicted_class", "risk_level"],
)

prediction_confidence = Histogram(
    "radiologyai_prediction_confidence",
    "Distribution of model confidence for completed predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

inference_latency_ms = Histogram(
    "radiologyai_inference_latency_ms",
    "Inference latency in milliseconds",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)

upload_validation_failures_total = Counter(
    "radiologyai_upload_validation_failures_total",
    "Total number of upload validation failures",
    ["reason"],
)

model_loaded_gauge = Gauge(
    "radiologyai_model_loaded",
    "Whether the model is loaded and ready",
)

model_info = Gauge(
    "radiologyai_model_info",
    "Static model metadata",
    ["model_name", "framework", "version"],
)

backend_info = Gauge(
    "radiologyai_backend_info",
    "Backend application metadata",
    ["app_name", "version"],
)

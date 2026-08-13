import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.inference.predict import load_model, predict, preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("catsdogs_api")

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.pt"))

app = FastAPI(title="Cats vs Dogs Classifier")
model = load_model(str(MODEL_PATH) if MODEL_PATH.exists() else None)

REQUEST_COUNT = Counter("request_count", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds", ["endpoint"])


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(endpoint=endpoint, status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    logger.info(
        json.dumps(
            {
                "endpoint": endpoint,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": round(duration * 1000, 2),
            }
        )
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        tensor = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    label, probs = predict(model, tensor)
    logger.info(json.dumps({"prediction": label, "probabilities": probs}))
    return {"label": label, "probabilities": probs}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

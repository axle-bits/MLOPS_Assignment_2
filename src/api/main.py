import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference.predict import load_model, predict, preprocess_image

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.pt"))

app = FastAPI(title="Cats vs Dogs Classifier")
model = load_model(str(MODEL_PATH) if MODEL_PATH.exists() else None)


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
    return {"label": label, "probabilities": probs}

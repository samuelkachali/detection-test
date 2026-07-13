from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from core import load_all_assets, predict, ModelLoadError

app = FastAPI(title="Crop Disease Detection API", version="1.0.0")

models = None

@app.on_event("startup")
def load_models_event():
    global models
    try:
        models = load_all_assets()
    except ModelLoadError as e:
        raise RuntimeError(str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    global models
    if models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    result = predict(image, models)
    return JSONResponse(content=result)

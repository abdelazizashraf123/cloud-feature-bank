"""FastAPI server for the cloud feature bank — multi-model version."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.services import FeatureBankService, MODEL_REGISTRY


service = FeatureBankService()

app = FastAPI(
    title="Cloud Feature Bank API",
    description="DINOv2 features + FAISS retrieval for CIFAR-100. Multi-model.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["info"])
def health():
    return service.health_info()


@app.get("/", tags=["info"])
def root():
    return {
        "message": "Cloud Feature Bank API (multi-model)",
        "docs": "/docs",
        "health": "/health",
        "available_models": list(MODEL_REGISTRY.keys()),
    }


@app.post("/extract", tags=["features"])
async def extract(
    file: UploadFile = File(...),
    model: str = Query(default="vits14", description="Model: vits14, vitb14, or vitl14"),
):
    """Extract DINOv2 features. Use ?model=vits14|vitb14|vitl14 to select."""
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        features, infer_ms = service.extract_features(image, model_name=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "model": model,
        "feature_dim": features.shape[1],
        "features": features[0].tolist(),
        "inference_time_ms": round(infer_ms, 2),
    }


@app.post("/retrieve", tags=["retrieval"])
async def retrieve(
    file: UploadFile = File(...),
    k: int = Query(default=20, ge=1, le=200, description="Number of neighbors"),
    model: str = Query(default="vits14", description="Model: vits14, vitb14, or vitl14"),
):
    """Retrieve top-K nearest neighbors using a specific DINOv2 variant."""
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        return service.retrieve(image, k=k, model_name=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
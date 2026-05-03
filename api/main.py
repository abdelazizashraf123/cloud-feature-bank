"""FastAPI server for the cloud feature bank."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.services import FeatureBankService
from api.models import ExtractResponse, RetrieveResponse, HealthResponse


# Initialize service ONCE at startup (loads model + index)
service = FeatureBankService(data_dir="local_data")

app = FastAPI(
    title="Cloud Feature Bank API",
    description="DINOv2 features + FAISS retrieval for vision datasets",
    version="0.1.0",
)

# Allow requests from any origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    """Health check + service info."""
    return service.health_info()


@app.get("/", tags=["info"])
def root():
    """Friendly landing message."""
    return {
        "message": "Cloud Feature Bank API",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/extract", response_model=ExtractResponse, tags=["features"])
async def extract(file: UploadFile = File(...)):
    """Extract DINOv2 features for an uploaded image."""
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    features, infer_ms = service.extract_features(image)
    return {
        "feature_dim": features.shape[1],
        "features": features[0].tolist(),
        "inference_time_ms": round(infer_ms, 2),
    }


@app.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
async def retrieve(
    file: UploadFile = File(...),
    k: int = Query(default=20, ge=1, le=200, description="Number of neighbors"),
):
    """Extract features and return top-K nearest neighbors with labels."""
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    return service.retrieve(image, k=k)
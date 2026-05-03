"""Pydantic models for API request/response validation."""

from typing import List
from pydantic import BaseModel, Field


class Neighbor(BaseModel):
    """A single retrieved neighbor."""
    index: int = Field(..., description="Position of the image in the dataset")
    label: int = Field(..., description="Class label (integer)")
    class_name: str = Field(..., description="Human-readable class name")
    similarity: float = Field(..., description="Cosine similarity score")


class ExtractResponse(BaseModel):
    """Response for /extract endpoint."""
    feature_dim: int
    features: List[float]
    inference_time_ms: float


class RetrieveResponse(BaseModel):
    """Response for /retrieve endpoint."""
    feature_dim: int
    features: List[float]
    neighbors: List[Neighbor]
    predicted_label: int
    predicted_class: str
    inference_time_ms: float
    search_time_ms: float


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str
    model: str
    dataset: str
    num_indexed: int
    feature_dim: int
    device: str
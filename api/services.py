"""Service layer: DINOv2 feature extraction + FAISS retrieval.

Supports multiple model sizes (ViT-S/B/L) loaded simultaneously.
Model selection is per-request via the 'model' parameter.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import faiss
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import hf_hub_download


DATASET_REPO = os.getenv("DATASET_REPO", "abdelazizashraf/cifar100-dinov2-features")

# Model registry: short_name -> HF model id
MODEL_REGISTRY = {
    "vits14": "facebook/dinov2-small",
    "vitb14": "facebook/dinov2-base",
    "vitl14": "facebook/dinov2-large",
}

DEFAULT_MODEL = "vits14"


class FeatureBankService:
    """Loads all DINOv2 models + FAISS indexes once, serves queries."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[init] Using device: {self.device}")

        # Storage for each model variant
        self.models = {}       # short_name -> (processor, model)
        self.indexes = {}      # short_name -> faiss index
        self.metadata = {}     # short_name -> dict
        self.train_labels = {} # short_name -> numpy array

        # Load all 3 variants
        for short_name, hf_model_id in MODEL_REGISTRY.items():
            self._load_variant(short_name, hf_model_id)

        # Use ViT-S metadata as "default" info source
        self.default_model = DEFAULT_MODEL
        print(f"[init] All variants loaded. Default: {self.default_model}")
        print("[init] Service ready.")

    def _load_variant(self, short_name: str, hf_model_id: str):
        """Load one model variant: model weights + FAISS index + labels."""
        print(f"\n[init] Loading variant: {short_name} ({hf_model_id})")
        t0 = time.time()

        # Download files
        index_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=f"cifar100_{short_name}.faiss",
            repo_type="dataset",
        )
        meta_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=f"cifar100_{short_name}_metadata.json",
            repo_type="dataset",
        )
        labels_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=f"cifar100_{short_name}_train_labels.npy",
            repo_type="dataset",
        )

        # Load metadata + labels
        with open(meta_path) as f:
            self.metadata[short_name] = json.load(f)
        self.train_labels[short_name] = np.load(labels_path)

        # Load FAISS
        self.indexes[short_name] = faiss.read_index(str(index_path))

        # Load model
        processor = AutoImageProcessor.from_pretrained(hf_model_id)
        model = AutoModel.from_pretrained(hf_model_id).to(self.device).eval()
        self.models[short_name] = (processor, model)

        dim = self.metadata[short_name]["feature_dim"]
        n = self.indexes[short_name].ntotal
        print(f"[init] {short_name}: dim={dim}, vectors={n}, "
              f"loaded in {time.time()-t0:.1f}s")

    def extract_features(self, image: Image.Image, model_name: str) -> tuple[np.ndarray, float]:
        """Run a specific DINOv2 variant on a PIL image."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. "
                             f"Available: {list(self.models.keys())}")
        processor, model = self.models[model_name]

        t0 = time.time()
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        elapsed_ms = (time.time() - t0) * 1000
        return features, elapsed_ms

    def retrieve(self, image: Image.Image, k: int = 20,
                 model_name: str = None) -> dict:
        """Extract features with chosen model and search its FAISS index."""
        if model_name is None:
            model_name = self.default_model
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. "
                             f"Available: {list(self.models.keys())}")

        # 1. Feature extraction
        features, infer_ms = self.extract_features(image, model_name)

        # 2. Normalize
        features_norm = features.copy()
        faiss.normalize_L2(features_norm)

        # 3. FAISS search (in the matching index)
        index = self.indexes[model_name]
        train_labels = self.train_labels[model_name]
        class_names = self.metadata[model_name]["class_names"]

        t0 = time.time()
        similarities, indices = index.search(features_norm, k)
        search_ms = (time.time() - t0) * 1000

        # 4. Build neighbor list
        neighbors = []
        for idx, sim in zip(indices[0], similarities[0]):
            label = int(train_labels[idx])
            neighbors.append({
                "index": int(idx),
                "label": label,
                "class_name": class_names[label],
                "similarity": float(sim),
            })

        # 5. Majority vote prediction
        neighbor_labels = [n["label"] for n in neighbors]
        predicted_label = int(np.bincount(neighbor_labels).argmax())

        return {
            "model": model_name,
            "feature_dim": features.shape[1],
            "features": features[0].tolist(),
            "neighbors": neighbors,
            "predicted_label": predicted_label,
            "predicted_class": class_names[predicted_label],
            "inference_time_ms": round(infer_ms, 2),
            "search_time_ms": round(search_ms, 2),
        }

    def health_info(self) -> dict:
        """Service health info."""
        variants_info = {}
        for short_name in self.models:
            variants_info[short_name] = {
                "model_id": MODEL_REGISTRY[short_name],
                "feature_dim": self.metadata[short_name]["feature_dim"],
                "num_indexed": int(self.indexes[short_name].ntotal),
            }
        return {
            "status": "ok",
            "device": self.device,
            "default_model": self.default_model,
            "available_models": list(self.models.keys()),
            "variants": variants_info,
            "dataset": "CIFAR-100",
        }
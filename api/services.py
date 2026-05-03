"""Service layer: DINOv2 feature extraction + FAISS retrieval."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import faiss
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class FeatureBankService:
    """Loads model and FAISS index once, serves queries."""

    def __init__(self, data_dir: str = "local_data"):
        self.data_dir = Path(data_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[init] Using device: {self.device}")
        self._load_metadata()
        self._load_model()
        self._load_index()
        print("[init] Service ready.")

    def _load_metadata(self):
        """Load metadata JSON and train labels."""
        meta_path = self.data_dir / "cifar100_vits14_metadata.json"
        labels_path = self.data_dir / "cifar100_vits14_train_labels.npy"

        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.train_labels = np.load(labels_path)
        self.class_names = self.metadata["class_names"]
        self.model_name = self.metadata["model"]
        self.feature_dim = self.metadata["feature_dim"]
        self.dataset_name = self.metadata["dataset"]

        print(f"[init] Metadata loaded: {self.dataset_name}, "
              f"{len(self.class_names)} classes, dim={self.feature_dim}")

    def _load_model(self):
        """Load DINOv2 model and processor."""
        print(f"[init] Loading model: {self.model_name}")
        t0 = time.time()
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        print(f"[init] Model loaded in {time.time()-t0:.1f}s")

    def _load_index(self):
        """Load FAISS index from disk."""
        index_path = self.data_dir / "cifar100_vits14.faiss"
        print(f"[init] Loading FAISS index from {index_path}")
        t0 = time.time()
        self.index = faiss.read_index(str(index_path))
        print(f"[init] Index loaded in {time.time()-t0:.2f}s, "
              f"{self.index.ntotal} vectors")

    def extract_features(self, image: Image.Image) -> tuple[np.ndarray, float]:
        """Run DINOv2 on a PIL image. Returns (features, time_ms)."""
        t0 = time.time()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # CLS token = global feature
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        elapsed_ms = (time.time() - t0) * 1000
        return features, elapsed_ms

    def retrieve(self, image: Image.Image, k: int = 20) -> dict:
        """Extract features and search for K nearest neighbors."""
        # 1. Feature extraction
        features, infer_ms = self.extract_features(image)

        # 2. Normalize for cosine similarity
        features_norm = features.copy()
        faiss.normalize_L2(features_norm)

        # 3. FAISS search
        t0 = time.time()
        similarities, indices = self.index.search(features_norm, k)
        search_ms = (time.time() - t0) * 1000

        # 4. Build neighbor list
        neighbors = []
        for idx, sim in zip(indices[0], similarities[0]):
            label = int(self.train_labels[idx])
            neighbors.append({
                "index": int(idx),
                "label": label,
                "class_name": self.class_names[label],
                "similarity": float(sim),
            })

        # 5. Predict via majority vote
        neighbor_labels = [n["label"] for n in neighbors]
        predicted_label = int(np.bincount(neighbor_labels).argmax())

        return {
            "feature_dim": features.shape[1],
            "features": features[0].tolist(),
            "neighbors": neighbors,
            "predicted_label": predicted_label,
            "predicted_class": self.class_names[predicted_label],
            "inference_time_ms": round(infer_ms, 2),
            "search_time_ms": round(search_ms, 2),
        }

    def health_info(self) -> dict:
        """Service health info."""
        return {
            "status": "ok",
            "model": self.model_name,
            "dataset": self.dataset_name,
            "num_indexed": int(self.index.ntotal),
            "feature_dim": int(self.feature_dim),
            "device": self.device,
        }
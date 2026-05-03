---
title: Cloud Feature Bank API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Cloud Feature Bank API

REST API serving precomputed DINOv2 features for CIFAR-100, indexed with FAISS for nearest-neighbor retrieval.

## Endpoints

- `GET /` — landing page
- `GET /health` — service info
- `GET /docs` — interactive API documentation (Swagger UI)
- `POST /extract` — extract DINOv2 features for an uploaded image
- `POST /retrieve?k=10` — extract features and return top-K nearest neighbors with labels

## Example

```bash
curl -X POST "https://abdelazizashraf-cloud-feature-bank.hf.space/retrieve?k=10" \
  -F "file=@your_image.jpg"
```

## Stack

- **DINOv2 ViT-S/14** — feature extractor (384-dim CLS token features)
- **FAISS IndexFlatIP** — exact cosine similarity search
- **FastAPI** — REST framework
- **CIFAR-100** — 50,000 indexed training images, 100 classes

## Performance

- kNN classification accuracy (K=20): **80.43%** on CIFAR-100 test set
- Inference latency on CPU (HF free tier): ~250-400 ms per request
- Inference latency on GPU (RTX 3060 baseline): ~63 ms per request

## Source

- **Code:** https://github.com/abdelazizashraf123/cloud-feature-bank
- **Features dataset:** https://huggingface.co/datasets/abdelazizashraf/cifar100-dinov2-features

## Authors

- Abdelaziz Hussein
- Shahzeen Ijaz Ahmad

Computer Engineering Department, Özyeğin University, Istanbul, Turkey
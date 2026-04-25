# Cloud-Based Feature Bank for Vision Datasets

A cloud service that stores precomputed DINOv2 features for vision datasets
and serves them through a REST API. Deployed on Hugging Face Spaces.

## Status
🚧 In development

## Stack
- DINOv2 (feature extractor)
- FAISS (vector index)
- FastAPI (REST server)
- Hugging Face Spaces (deployment)

## Project structure
- `notebooks/` — Feature extraction (run on Colab)
- `api/` — FastAPI server code
- `studies/` — Experiment results

## Authors
- Abdelaziz Hussein
- Shahzeen Ijaz Ahmad

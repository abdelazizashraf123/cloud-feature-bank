FROM python:3.11-slim

# Hugging Face Spaces runs as user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

# Install CPU-only PyTorch (HF Spaces free tier has no GPU)
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user \
        torch==2.5.0 torchvision==0.20.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy app code
COPY --chown=user . /app

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
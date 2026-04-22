FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    insightface onnxruntime \
    opencv-python-headless scipy scikit-learn \
    openai-whisper librosa decord \
    tqdm Pillow PyYAML huggingface_hub mss

# App code
COPY . .

# Env defaults (override via docker-compose or -e flags)
ENV ADB_PROFILE=cpu_only
ENV PYTHONPATH=/app:/app/third_party/rppg-toolbox:/app/third_party/faceforensics/classification:/app/third_party/latentsync

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

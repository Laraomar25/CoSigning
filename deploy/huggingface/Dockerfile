FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements.txt

# Download ViT once at build time so the first request is fast
RUN python -c "from transformers import ViTImageProcessor, ViTModel; ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'); ViTModel.from_pretrained('google/vit-base-patch16-224')"

COPY --chown=user backend backend
COPY --chown=user features_5.0/features_5.0/inference_package_5_0.pkl features_5.0/features_5.0/inference_package_5_0.pkl

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]

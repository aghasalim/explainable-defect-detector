# Runs the demo exactly as Hugging Face Spaces does: CPU-only, port 7860.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    # torch.hub would otherwise write to a read-only HOME on Spaces
    TORCH_HOME=/app/.torch

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Warm the backbone into the image so first request is not a 100 MB download
RUN python -c "from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights; \
    wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)"

COPY src/ ./src/
COPY models/ ./models/
COPY assets/ ./assets/
COPY app.py .

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:7860/_stcore/health')"

CMD ["streamlit", "run", "app.py"]

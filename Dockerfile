FROM python:3.11-slim

# Tesseract is a system dep for OCR; install it alongside Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY comas_triage ./comas_triage
COPY templates ./templates
COPY config.yaml ./config.yaml

EXPOSE 8000

# Default to serving the web UI. Override with --entrypoint for training.
CMD ["uvicorn", "comas_triage.app:app", "--host", "0.0.0.0", "--port", "8000"]

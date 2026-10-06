FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ENVIRONMENT=production \
    PREWARM_MODELS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libreoffice-writer \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.lock.txt /tmp/requirements.lock.txt
RUN python -m pip install --no-cache-dir --requirement /tmp/requirements.lock.txt \
    && rm /tmp/requirements.lock.txt

COPY backend/app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

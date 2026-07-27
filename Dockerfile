FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VT_DATA_DIR=/app/data \
    VT_OUTPUT_DIR=/app/data/outputs \
    VT_OPENVOICE_DIR=/app/checkpoints_v2

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements-gpu.txt
COPY . .
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "voice_translator.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

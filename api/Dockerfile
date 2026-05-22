FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/final_model.pkl \
    MODEL_METADATA_PATH=/app/handoff/model/model_metadata.json \
    MODEL_THRESHOLD=0.70 \
    API_VERSION=1.0.0 \
    MODEL_VERSION=sprint5-final-threshold-0.70

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY api/ /app/api/
COPY src/ /app/src/
COPY handoff/ /app/handoff/
COPY models/ /app/models/
COPY data/processed/test_final.csv /app/data/processed/test_final.csv

RUN useradd -m appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

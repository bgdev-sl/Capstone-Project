# Reproducible container for the local research API and dashboard.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    EDGE_IDS_ROOT=/app \
    EDGE_IDS_CONFIG=/app/configs/config.yaml

WORKDIR /app

# XGBoost uses libgomp on Linux; install only the runtime library needed for it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data models logs results

EXPOSE 8000 8501

# Compose overrides this command per service.
CMD ["uvicorn", "edge_ids.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libmariadb-dev \
        libmariadb-dev-compat \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m playwright install --with-deps chromium

COPY backend/ .
COPY gs_data /gs_data

EXPOSE 8003
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8003"]

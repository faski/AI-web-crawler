FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ .

EXPOSE 8004
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8004"]

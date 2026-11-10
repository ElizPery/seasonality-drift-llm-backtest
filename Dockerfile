# Monitoring service of stream B: FastAPI + the backtest engine (see src/api/main.py)
FROM python:3.12.16-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY src/ src/
# one thread per model fit, as in the experiments (the numbers stay the same)
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

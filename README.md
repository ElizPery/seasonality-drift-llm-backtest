## Repository Structure

```text
seasonality-drift-llm-mlops/
│
├── .github/workflows/          # CI/CD pipelines (GitHub Actions for pytest)
├── data/                       # Directory for datasets
│   ├── raw/                    # Raw data (Alibaba, NAB, UCI Electricity, TCPD)
│   └── processed/              # Processed and synthetically labeled series
│
├── notebooks/                  # Research Jupyter Notebooks for experiments
│   ├── 01_eda_and_stl.ipynb    # Seasonality analysis (STL decomposition)
│   └── 02_drift_injection.ipynb# Synthetic drift generator verification
│
├── src/                        # Core service source code
│   ├── __init__.py
│   ├── detector/               # Stream A: Drift detectors and decomposition
│   │   ├── __init__.py
│   │   ├── stl_detector.py     # STL decomposition + residuals
│   │   └── baseline_detectors.py# KS, PSI, ADWIN, Page-Hinkley (River / SciPy)
│   │
│   ├── recommendations/        # Stream B: LLM diagnostics and backtesting
│   │   ├── __init__.py
│   │   ├── llm_client.py       # Instructor + Pydantic v2 schemas for LLM
│   │   ├── catalog.py          # Remediation action catalog (Actions)
│   │   └── backtester.py       # Backtesting engine with block Bootstrap CI
│   │
│   ├── api/                    # MLOps microservice
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application (/detect, /recommendations endpoints)
│   │   └── schemas.py          # Pydantic schemas for API requests/responses
│   │
│   └── monitoring/             # Infrastructure integration
│       ├── __init__.py
│       └── metrics.py          # Prometheus metrics (prom_client)
│
├── tests/                      # Automated tests (Pytest)
│   ├── test_detector.py
│   └── test_backtester.py
│
├── docker-compose.yml          # Local deployment (FastAPI, MLflow, Prometheus, Grafana)
├── Dockerfile                  # FastAPI service containerization
├── requirements.txt            # Python 3.11+ dependencies
└── README.md                   # Project description and setup instructions
```
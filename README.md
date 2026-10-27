## Repository Structure

```text
    seasonality-drift-llm-mlops/
    │
    ├── .github/workflows/          # CI/CD pipelines (GitHub Actions for pytest)
    ├── data/                       # Directory for datasets
    │   ├── raw/                    # Raw data (Alibaba, UCI Electricity, TCPDBench)
    │   └── processed/              # Processed and synthetically labeled series
    │
    ├── notebooks/                  # Research Jupyter Notebooks for experiments
    │   └── 01_eda_and_baselines.ipynb   
    │
    ├── src/                        # Core service source code
    │   ├── __init__.py
    │   ├── detector/               # Stream A: Drift detectors and decomposition
    │   │   ├── data_loader.py          # Data loading and standardization (Alibaba, UCI, TCPDBench)
    │   │   ├── drift_generator.py      # STL decomposition + 3-level benchmark generation
    │   │   ├── baseline_detectors.py   # KS, PSI, ADWIN, Page-Hinkley, KSWIN (River / SciPy)
    │   │   ├── generate_benchmarks.py
    │   │   └── run_baseline_evaluation.py # Baseline metric runner
    │   │
    │   ├── recommendations/        # Stream B: LLM diagnostics and backtesting
    │   │   ├── __init__.py
    │   │   ├── llm_client.py       # Instructor + Pydantic v2 schemas for LLM
    │   │   ├── catalog.py          # Remediation action catalog
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


## Data Source & Setup

### 1. UCI ElectricityLoadDiagrams 2011–2014 (`LD2011_2014.txt`)

A classic benchmark time-series dataset containing electricity consumption data for 370 clients sampled at 15-minute intervals.

* **Official Source:** [UCI Machine Learning Repository - ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
* **Direct Download:** [UCI Zip Archive (249 MB)](https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip)

#### Setup Instructions:
1. Download and extract `electricityloaddiagrams20112014.zip`.
2. Move the unzipped `LD2011_2014.txt` file into your local project directory:
   ```path
   data/raw/LD2011_2014.txt
   ```
3. The `load_uci_electricity()` method in `DataLoader` will automatically read and standardize this file.


### 2. Alibaba Cluster Trace 2018 (`alibaba_cluster_trace.csv`)

The raw Alibaba trace is large and resource-intensive to parse from scratch. For faster setup, use the pre-aggregated and cleaned open-access slice provided on Zenodo.

* **Dataset Repository:** [DataCenter-Traces-Datasets on Zenodo](https://zenodo.org/records/14564935)
* **Target File:** `machine_usage_days_1_to_8_grouped_300_seconds.csv` (~200 KB)

#### Setup Instructions:
1. Navigate to the Zenodo dataset page linked above.
2. Download `machine_usage_days_1_to_8_grouped_300_seconds.csv`.
3. Save and rename the file into your local project directory:
```path
   data/raw/alibaba_cluster_trace.csv
   ```
4. The `load_alibaba_standard_trace()` method in `DataLoader` will automatically parse this aggregated CPU/memory usage trace.

### 3. TCPDBench Dataset (`jfk_passengers` / Level 3 Benchmark)

TCPDBench (Time Series Change Point Benchmarks) contains real-world time series datasets with various change points and conceptual drift, which serve as the Level 3 benchmark for validating drift detectors.

- **Official Repository:** [GitHub - TCPDBench/TCPDBench](https://github.com/TimeSeriesAI/TCPDBench)
- **Target File:** E.g., `jfk_passengers.csv` or specific series files from the benchmark repository.

#### Setup Instructions:
1. Download the required time series files from the TCPDBench repository.
2. Place them into your project's raw data directory:
   ```path
   data/raw/tcpd/
   ```
3. The loading method in `DataLoader` will automatically read and prepare the Level 3 series for detector evaluation.
## Repository Structure

```text
    seasonality-drift-llm-mlops/
    │
    ├── .github/workflows/          # CI/CD pipelines (GitHub Actions for pytest)
    ├── data/                       # Directory for datasets
    │   ├── raw/                    # Raw data (Alibaba, UCI Electricity, TCPDBench, SeisBench)
    │   └── processed/              # Processed and synthetically labeled series
    │
    ├── notebooks/                  # Research Jupyter Notebooks for experiments
    │   ├── 01_eda_and_baselines.ipynb
    │   ├── 02_residual_conformal_evaluation.ipynb # Stream A: Residual conformal evaluation & visualization
    │   └── 02_stream_b_baselines.ipynb  # Stream B: strategies, oracles, examples
    │
    ├── src/                        # Core service source code
    │   ├── __init__.py
    │   ├── data/                   # Shared by both streams: loaders, the standard frame, the manifest
    │   │   ├── uci.py, entsoe.py, seismic_noise.py, alibaba.py, tcpd.py, iquique.py   # one loader per source
    │   │   ├── frames.py           # standard frame (timestamp, value), series_table, write_series
    │   │   ├── loader.py           # DataLoader class interface used by stream A
    │   │   └── export.py           # python -m src.data.export: the stream B series as CSV + manifest.csv
    │   ├── detector/               # Stream A: Drift detectors and decomposition
    │   │   ├── data_loader.py          # Kept import path of DataLoader (the loaders live in src/data)
    │   │   ├── drift_generator.py      # STL decomposition + 3-level benchmark generation
    │   │   ├── baseline_detectors.py   # KS, PSI, ADWIN, Page-Hinkley, KSWIN (River / SciPy)
    │   │   ├── residual_detector.py    # STL decomposition + adaptive conformal intervals on residuals
    │   │   ├── generate_benchmarks.py
    │   │   ├── run_baseline_evaluation.py # Baseline metric runner 
    │   │   └── run_residual_detector_evaluation.py # Residual conformal evaluation runner
    │   │
    │   ├── recommendations/        # Stream B: LLM diagnostics and backtesting
    │   │   ├── forecast.py         # LightGBM day-ahead model, seasonal naive, MASE
    │   │   ├── catalog.py          # Remediation action catalog
    │   │   ├── backtester.py       # Strategies, hindsight oracles, block bootstrap CI of the MASE gain
    │   │   ├── synthetic.py        # Synthetic series from UCI clients (MSTL control series + injected changes)
    │   │   ├── data_sources.py     # ENTSO-E and seismic noise series with features and labels (loaders in src/data)
    │   │   ├── plots.py            # Figures for the stream B notebook
    │   │   ├── run_backtest.py     # Stream B runner
    │   │   └── llm_client.py       # Instructor + Pydantic v2 schemas for LLM
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

### 4. SeisBench Iquique Seismic Trace Dataset

SeisBench provides standardized seismic datasets for machine learning applications in seismology. The Iquique seismic trace dataset (focusing on P-wave arrivals with ground-truth change points) is used to evaluate drift and abrupt change detection under high-noise geophysical signal conditions.

- **Official Source / Toolkit:** [GitHub - SeisBench/seisbench](https://github.com/seisbench/seisbench)
- **Dataset Focus:** Iquique regional seismic network, P-wave picks, and event-based change points.

#### Setup Instructions:
1. The corresponding loading method in `DataLoader` will automatically ingest the seismic waveform slice, parse the ground-truth change points and prepare it for seasonal-trend drift evaluation.

### 5. ENTSO-E National Load (Open Power System Data), Stream B

Hourly national electricity load of 32 European countries, 2015 to September 2020, from the ENTSO-E Transparency
Platform as packaged by Open Power System Data. Stream B uses 8 countries (IT, PL, ES, NL, PT, FR, DE, GB) and the
date of the first national COVID-19 measures in March 2020 as the moment of change.

- **Official Source:** [Open Power System Data - Time series](https://data.open-power-system-data.org/time_series/2020-10-06)
- **Target File:** `time_series_60min_singleindex.csv` (124 MB)

#### Setup Instructions:
```bash
mkdir -p data/raw/opsd
curl -L -o data/raw/opsd/time_series_60min_singleindex.csv \
  https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv
```

### 6. Seismic Noise During the 2020 Lockdowns (Lecocq et al., 2020), Stream B

Displacement RMS every 15 minutes at seismic stations, November 2019 to May 2020, from the companion repository of
Lecocq et al. (2020), Science 369(6509):1338-1343. The 4-14 Hz band is dominated by human activity (traffic, industry),
so it has clear daily and weekly cycles. Stream B uses 15 stations; the moment of change is the date of the first
measures at each station from the paper's Table S1. This is a different dataset from the SeisBench Iquique records above.

- **Official Source:** [GitHub - ThomasLecocq/2020_Science_GlobalQuieting](https://github.com/ThomasLecocq/2020_Science_GlobalQuieting)

#### Setup Instructions:
```bash
R=https://raw.githubusercontent.com/ThomasLecocq/2020_Science_GlobalQuieting/master
mkdir -p data/raw/seismic_noise
curl -L -o data/raw/seismic_noise/lockdown_stations.csv "$R/Supplementary%20Files/Table%20S1%20-%20Lockdown%20stations.csv"
for f in BE.UCC..HHZ IV.MILN..HHZ IV.FIR..HHZ FR.CURIE.00.HHZ GB.EDI..HHZ NL.ALK1..HGZ GE.STU..HHZ GR.BUG..HHZ \
         CA.ICJA..HHZ CH.SZUZ..HGZ OE.VIE..HHZ LD.CPNY..BHZ AM.R091F.00.EHZ AM.R9F13.00.EHZ AM.S0C34.00.EHZ; do
  curl -L -o "data/raw/seismic_noise/$f.csv" "$R/PaperZero_RMS/$f.csv"
done
```


## Shared data loading (`src/data`)

Both streams read their data through `src/data`. Every loader returns the same standard frame: a DataFrame
with the columns `timestamp` (start of the interval) and `value` (float64), one row per point in time order.
`DataLoader` (`src/data/loader.py`, also importable from `src/detector/data_loader.py`) keeps the class
interface of the stream A scripts and notebooks; `load_uci_electricity` now reads the cleaned hourly client
table of `src/data/uci.py` (timestamps moved to the interval start, DST hours fixed, cached in
`data/processed/uci_hourly.parquet`).

The series of stream B (synthetic UCI, ENTSO-E, seismic noise) can be written as standard frames for the
detectors of stream A:

```bash
python -m src.data.export            # data/processed/series/: one CSV per series + manifest.csv
```

`manifest.csv` has one row per series: `id` (`source/series/scenario`), `file`, the labels (`source`,
`series`, `scenario`, `client`, `seed`, `magnitude`), the length `n`, `start`, and the deployment and
change points both as positions (`t_deploy`, `t_change`, `t_stop`) and as timestamps (`deploy_at`,
`change_at`). Detector alarms on these series are matched to the backtest by `id` and `timestamp`.

## Stream A: Residual Conformal Drift Detection and Evaluation (`src/detector`)

To solve the alert fatigue problem common in seasonal time series (where traditional batch tests like PSI and Wasserstein trigger massive false alarms due to natural seasonality), Stream A implements a **seasonality-aware dual-layer detector**:

- **`residual_detector.py`**: Performs STL decomposition to separate trend, seasonality, and residuals. It computes **adaptive conformal intervals** on the residuals to monitor two independent anomaly dimensions:
  1. *Absolute residual deviation* (detecting sudden level shifts or mean changes).
  2. *Local Interquartile Range (IQR) dispersion* (detecting volatility and amplitude shifts).
- **`run_residual_detector_evaluation.py`**: Executes the evaluation runner across the 3-level benchmark suite (Level 1 clean block-bootstrap series for False Alarm Rate, Level 2 semi-synthetic injected drifts across $0.5\sigma$ to $3\sigma$ magnitudes, and Level 3 real-world consensus datasets).
- **`notebooks/02_residual_conformal_evaluation.ipynb`**: Interactive research notebook containing comparative visualizations, precision-recall breakdowns, and detection delay curves against traditional streaming and batch baselines (ADWIN, KS, PSI, Page-Hinkley, KSWIN, Wasserstein).


## Stream B: Adaptation Strategies and Backtest (`src/recommendations`)

A deployed LightGBM model forecasts every day the next 24 hours (all features use values at least 24 hours old:
hour, weekday, lags 24/48/168/336, lagged daily and weekly means; yearly terms for the national load). A strategy
decides when the model is changed and how:

- `nothing`: the model trained at deployment is never changed;
- `always`: retrain on all history every day;
- `scheduled`: retrain on all history once a week;
- `scheduled_gate`: once a week, retrain on all history only if the acceptance rule confirms the gain;
- `catalog_gate`: once a week, check every catalog action with the acceptance rule and apply the one with the highest
  lower bound of the gain (or keep the model if none passes);
- `oracle_weekly`: once a week, apply the catalog action with the lowest error on the following week, chosen in
  hindsight (the reference for LLM recommendations);
- `oracle`: the same every day with a one-day lookahead (the ideal ceiling).

The oracles use future data, so they cannot be deployed; they only bound what any policy can reach. Action catalog
(`catalog.py`): do nothing, retrain on all history, retrain on the last 8 weeks, shorten the window to 2 weeks,
reweight recent data (half-life 1 week), add a seasonal feature (4-week hour-of-week profile), recalibrate the forecast
level on the last week.

Metrics: MASE (scaled by the weekly seasonal naive error on the training data) before and after the change, regret
against the ideal oracle, number of retrainings, and the gain over `nothing` with a 95% moving block bootstrap
interval over days (blocks of 7 days, 2000 resamples).

Acceptance rule (`check_action` in `backtester.py`): at a decision point the recommended action is replayed on the
last 2 weeks with a weekly rolling origin (its recipe is applied to the data before each origin and forecasts the
following week) and compared with what the deployed model actually forecast there. The action is applied only if the
lower bound of the 95% block bootstrap interval of the gain (blocks of 3 days) is above delta = 0. Only data before
the decision point is used.

LLM recommendations (`llm_client.py`): at a decision point the advisor gets diagnostics computed from the data before
it (weekly errors of the deployed model, bias, level, daily amplitude and noise changes, weeks since the last change)
and answers with one catalog action and a short reason (Pydantic schema, validated by Instructor in JSON mode).
Strategies `llm` (applied as proposed) and `llm_gate` (applied only if the acceptance rule confirms it); `rules` and
`rules_gate` use fixed rules on the same diagnostics. Every LLM answer is stored in
`data/llm_answers/` (kept in git), so a rerun reads the stored answers and needs no API key. The key is read from
`~/.config/diploma/anthropic.env` (one line `ANTHROPIC_API_KEY=...`, another path can be set in `LLM_KEY_FILE`).

Decisions on detector alarms (`alarms.py`, the link with stream A): the strategies `alarm_retrain`, `alarm_gate`,
`alarm_catalog_gate`, `alarm_oracle`, `alarm_rules`, `alarm_rules_gate`, `alarm_llm` and `alarm_llm_gate` take the
same decisions as their weekly counterparts, but at the first origin after an alarm of a drift detector, then ignore
alarms for 7 days (`cooldown_days`). They run once per detector in an alerts file: one row per alarm with `id`
(`source/series/scenario` as in the manifest of `src/data`), `method` and `timestamp` (or `index`); detector
alarms are matched to the series by `id` and `timestamp`. Every row of `runs.csv` also reports `alarms` (alarms after
deployment), `acts_pre` (model changes before the change of the series, all of them on a series without one: the cost
of false alarms or of a fixed calendar) and `reaction_h` (hours from the change to the first model change after it).
Until the detectors of stream A deliver their alarms, two stand-in detectors run on the same series, both using only
the past: `profile_conformal` (the residual of each hour from the median of the same hour of the last 4 weeks, a
conformal threshold on the last 2 weeks of residuals, two consecutive violations, one-day cooldown) and `adwin`
(ADWIN on the same residuals). Their sensitivity is calibrated on the 60 control series to one false alarm per 14 days;
`data/alerts/detectors.csv` records the chosen settings and `data/alerts/alerts.csv` their alarms (kept in git).

Data: 240 synthetic series (20 UCI clients x 3 seeds x {no change, level shift 2 sigma, one-week ramp 2 sigma,
noise x2}, 24 weeks, deployment at week 10, change at week 13), the national load of 8 countries (dataset 5,
deployment 6 January 2020) and the seismic noise level at 15 stations (dataset 6, hourly log10 on the local clock,
deployment 3 February 2020).

#### Run:
```bash
export OMP_NUM_THREADS=1                                     # every process trains its own models
python -m src.recommendations.run_backtest --tune            # base model validation (not the test period)
python -m src.recommendations.run_backtest                   # all sources, about 40 minutes on 16 cores
python -m src.recommendations.run_backtest --sources seismic # recompute one source, keep the other rows
python -m src.recommendations.run_backtest --advisors rules  # add the advised strategies (rules, llm) to runs.csv
python -m src.recommendations.alarms                         # stand-in detectors: calibrate, write data/alerts/alerts.csv
python -m src.recommendations.run_backtest --alerts data/alerts/alerts.csv --advisors rules
                                                             # add the strategies that review on alarms, per detector
python -m pytest tests                                       # tests
```

Results are written to `data/processed/stream_b/` (`runs.csv`, `summary.csv`, `oracle_picks.csv`, `tune.csv`,
`example_*.csv`); tables and figures are in `notebooks/02_stream_b_baselines.ipynb`. The published results were
computed with Python 3.12, numpy 2.2.6, pandas 2.3.3, statsmodels 0.15.0 and lightgbm 4.7.0.

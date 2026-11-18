#!/usr/bin/env bash
# Full pipeline of the project, from the raw data to the Grafana dashboard.
# Run from the root of the repository (branch main):
#   bash run_pipeline.sh         everything: data, series, tests, backtest, service
#   bash run_pipeline.sh demo    data, series and the service only (no backtest)
#   bash run_pipeline.sh data    only download the data
# Needs Python 3.12, Docker, curl and unzip.
set -euo pipefail
MODE="${1:-all}"

# --- data of stream B (about 400 MB, files that exist are not downloaded again) ---
mkdir -p data/raw/opsd data/raw/seismic_noise
if [ ! -f data/raw/LD2011_2014.txt ]; then
  UCI=https://archive.ics.uci.edu/static/public/321
  curl -L -o data/raw/uci.zip "$UCI/electricityloaddiagrams20112014.zip"
  unzip -o -j data/raw/uci.zip LD2011_2014.txt -d data/raw
  rm data/raw/uci.zip
fi
OPSD=https://data.open-power-system-data.org/time_series/2020-10-06
LOAD=time_series_60min_singleindex.csv
[ -f "data/raw/opsd/$LOAD" ] || curl -L -o "data/raw/opsd/$LOAD" "$OPSD/$LOAD"
R=https://raw.githubusercontent.com/ThomasLecocq/2020_Science_GlobalQuieting/master
S1="Supplementary%20Files/Table%20S1%20-%20Lockdown%20stations.csv"
[ -f data/raw/seismic_noise/lockdown_stations.csv ] ||
  curl -L -o data/raw/seismic_noise/lockdown_stations.csv "$R/$S1"
for f in BE.UCC..HHZ IV.MILN..HHZ IV.FIR..HHZ FR.CURIE.00.HHZ GB.EDI..HHZ \
         NL.ALK1..HGZ GE.STU..HHZ GR.BUG..HHZ CA.ICJA..HHZ CH.SZUZ..HGZ \
         OE.VIE..HHZ LD.CPNY..BHZ AM.R091F.00.EHZ AM.R9F13.00.EHZ AM.S0C34.00.EHZ; do
  [ -f "data/raw/seismic_noise/$f.csv" ] ||
    curl -L -o "data/raw/seismic_noise/$f.csv" "$R/PaperZero_RMS/$f.csv"
done
[ "$MODE" = data ] && exit 0

# --- environment and the series in the common format ---
[ -d .venv ] || python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data.export

# --- tests and the backtest (about an hour) ---
if [ "$MODE" = all ]; then
  python -m pytest tests
  export OMP_NUM_THREADS=1
  python -m src.recommendations.run_backtest
  python -m src.recommendations.run_backtest --advisors rules
  python -m src.recommendations.run_backtest --alerts data/alerts/alerts.csv --advisors rules
fi

# --- service: API, PostgreSQL, Prometheus, Grafana; one session on the Milan station ---
docker compose up --build -d
until curl -sf localhost:8000/health > /dev/null; do sleep 2; done
SESSION='{"series_id": "seismic/IV.MILN/lockdown_2020", "strategy": "alarm_rules_gate",
          "detector": "residual_conformal", "speed": 12}'
curl -s -X POST localhost:8000/sessions -H 'content-type: application/json' -d "$SESSION"
echo
echo "Open the grafana_url from the answer above. API: http://localhost:8000/docs"

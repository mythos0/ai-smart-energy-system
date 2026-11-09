# AI-Based Smart Electricity Consumption Prediction & Anomaly Detection System

An end-to-end machine learning system for **day-ahead household electricity
forecasting** and **anomaly detection**, built on the
[UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
(2.08M minute-level readings, Dec 2006 – Nov 2010), with a published-quality
research paper, reproducible training pipeline (Kaggle notebook), and a
**deployable web application** (Flask + Chart.js, Render-ready).

---

## Results at a glance

**Day-ahead forecasting** (strict protocol: every feature ≥ 24 h old; test = Apr–Nov 2010, 5,554 h):

| Model | RMSE (kW) | MAE (kW) | MAPE | R² | Skill vs naive |
|---|---|---|---|---|---|
| **XGBoost** (best) | **0.5875** | 0.4328 | 64.9% | **0.318** | **+22.4%** |
| LSTM | 0.5920 | **0.4213** | **55.9%** | 0.307 | +21.8% |
| LightGBM | 0.5931 | 0.4354 | 65.9% | 0.304 | +21.6% |
| Random Forest | 0.5937 | 0.4400 | 68.1% | 0.303 | +21.5% |
| Ridge | 0.6060 | 0.4513 | 65.4% | 0.274 | +19.9% |
| Seasonal naive 24 h | 0.7568 | 0.5101 | 66.0% | −0.13 | 0 |
| Holt-Winters | 0.7860 | 0.5949 | 92.4% | −0.22 | −3.9% |

**Anomaly detection** (2% synthetic injection: spikes / level shifts / dropouts, contaminated features):

| Detector | F1 | ROC-AUC | Spike recall | Shift recall | Dropout recall |
|---|---|---|---|---|---|
| **Residual z-score (XGB)** (best) | **0.482** | **0.885** | 0.932 | 0.333 | 0.057 |
| LSTM autoencoder | 0.402 | 0.721 | 0.977 | 0.061 | 0.000 |
| One-Class SVM | 0.232 | 0.726 | 0.523 | 0.091 | 0.000 |
| Isolation Forest | 0.196 | 0.731 | 0.409 | 0.121 | 0.000 |

Key findings: gradient-boosted trees beat deep learning on this tabular task;
household-level day-ahead R² tops out near 0.32 (behavioural stochasticity);
**all detectors catch spikes but miss dropouts** — the paper's central open problem.

---

## Repository structure

```
├── paper/                 # Research paper (LaTeX source + final 11-page PDF + cover)
│   ├── main.tex           # single-column journal-style manuscript
│   ├── refs.bib           # verified bibliography (natbib)
│   └── final_paper.pdf    # compiled paper (cover + 10 pages)
├── figures/               # All publication figures (vector PDF) + web screenshots
├── scripts/               # Full pipeline (run in order)
│   ├── 01_prepare_data.py     # download → clean → hourly aggregation → splits
│   ├── 02_train_forecast.py   # 8 forecasting models (resumable, checkpointed)
│   ├── 03_anomaly.py          # injection benchmark + 4 detectors + case study
│   ├── 04_figures.py          # all publication figures
│   ├── 05_test_web.py         # web app end-to-end tests + screenshots
│   └── 06_build_notebook.py   # builds the Kaggle notebook
├── notebooks/
│   └── training_pipeline.ipynb   # reproducible training (pushed to Kaggle)
├── models/                # trained artefacts
│   ├── xgboost.json       # best forecaster (used by the web app)
│   ├── lightgbm.joblib / ridge.joblib
│   ├── lstm_forecaster.pt / lstm_autoencoder.pt
│   └── random_forest_compressed.joblib
├── results/               # metrics, test predictions, anomaly scores, case study
├── data/                  # hourly_full.csv (aggregated); raw download scripted
└── web/                   # deployable Flask application
    ├── app.py             # dashboard + REST API
    ├── templates/         # Chart.js UI (forecast / anomalies / models / API)
    ├── model/             # serving artefacts (xgboost.json + data + metrics)
    ├── requirements.txt
    └── render.yaml        # one-click Render blueprint
```

## Reproduce

```bash
# 1. data (downloads 20 MB zip from UCI, cleans, aggregates)
python scripts/01_prepare_data.py

# 2. forecasting benchmark (~15 min CPU, resumable)
python scripts/02_train_forecast.py

# 3. anomaly detection benchmark (~10 min CPU)
python scripts/03_anomaly.py

# 4. figures
python scripts/04_figures.py
```

Or run everything in the cloud:
**[Kaggle notebook — ai-smart-energy-system-training-pipeline](https://www.kaggle.com/code/tamim6/ai-smart-energy-system-training-pipeline)**
(attaches `uciml/electric-power-consumption-data-set`, no setup required).

## Deploy the web app (Render)

The web app serves the XGBoost forecaster + residual anomaly detector:

- **Forecast tab** — backtest any day of the test period (24 h ahead, ±1.96σ band)
  or paste ≥216 hourly kW readings for a **live forecast**
- **Anomalies tab** — z-scores, flagged hours, strongest real events
- **Models tab** — both benchmark leaderboards
- **REST API** — `/api/forecast`, `/api/anomaly`, `/api/history`, `/api/models`, `/api/health`

[![Deploy to Render](https://render.com/images/deploy-button.svg)](https://dashboard.render.com/select-repo?type=web&repo=https%3A%2F%2Fgithub.com%2Fmythos0%2Fai-smart-energy-system)

**Option A — Blueprint (recommended, fully automatic).** In the Render dashboard:
**New → Blueprint** → select this repo → **Apply**. The repo-root `render.yaml`
(service `rootDir: web`) configures the build, start command, and health check
automatically. Free tier (512 MB RAM) is sufficient — the app loads only a 0.8 MB
model plus 5 MB of data and needs no GPU/DL runtime.

**Option B — New Web Service (manual).** Render dashboard → **New → Web Service** →
paste the public repo URL, then set:

| Setting | Value |
| --- | --- |
| Root directory | `web` |
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn app:app --workers 1 --threads 4 --timeout 120` |
| Health check path | `/api/health` |
| Instance type | Free |

Local run:

```bash
cd web
pip install -r requirements.txt
python app.py          # http://localhost:5000
```

API example:

```bash
curl -X POST http://localhost:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"values": [1.24, 0.98, 0.87, ... ]}'   # >= 216 hourly kW values
```

## Methodology (short version)

- **Task:** predict all 24 hourly kW values of day *d* using only information older
  than 23:00 of day *d−1* (calendar + lag-24/48/168 h + rolling stats, all shifted ≥ 24 h).
- **Splits:** chronological — train to Jun 2009, validation Jul 2009–Mar 2010,
  test Apr–Nov 2010 (never seen during model development).
- **Anomaly protocol:** inject 2% of test hours (spikes ×2.5–4, level shifts +60–150%,
  dropouts → 0.02–0.10 kW), recompute lag features from the contaminated series,
  and score point-wise detection at the prevalence-matched operating point.
- **Full details:** see `paper/final_paper.pdf`.

## Citation

```bibtex
@misc{hossen2026smartenergy,
  author = {Tamim Hossen, MD and Sarkar, Md Emon and Hasan, MD. Montasir and
            Ratul, MD. Sheratul Muntahar and Nisha, Afrina},
  title  = {AI-Based Smart Electricity Consumption Prediction and Anomaly Detection:
            A Benchmark Study and Deployable Web System},
  year   = {2026},
  howpublished = {Department of CSE, Dhaka International University},
  url    = {https://github.com/mythos0/ai-smart-energy-system}
}
```

Dataset citation: Hebrail, G. & Berard, A. (2012). *Individual Household Electric
Power Consumption*. UCI Machine Learning Repository.

## License

MIT (code). The dataset is redistribution-permitted for research use (see UCI page).

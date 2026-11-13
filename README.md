# AI-Based Smart Electricity Consumption Prediction & Anomaly Detection System

Final-year project — Department of CSE, Dhaka International University.

**The user enters their last 4 monthly electricity bills (kWh). The web app instantly shows
the next month's predicted usage, the predicted bill in BDT, and an anomaly check on the
latest month — all on one single screen, with no scrolling, on mobile and desktop.**

**Live app:** https://ai-smart-energy-system-w4tx.onrender.com

## How it works

| Step | What happens |
| --- | --- |
| 1. User input | The user types the units (kWh) of their last 4 monthly bills and their sanctioned load (kW) |
| 2. Prediction | A scale-free monthly model predicts next month's usage from the last 3 months |
| 3. Bill in BDT | The official Bangladesh tariff (BERC 2026, residential LT-A) is applied to the predicted usage: lifeline + slabs + ৳42/kW demand charge + 5% VAT |
| 4. Anomaly check | The latest month is compared with the model's expectation; a residual z-score beyond ±3 is flagged as unusual HIGH or LOW |

The monthly model is **scale-free**: it works on ratios to the 3-month average, so the same
model predicts for a single home, a feeder, or the whole city of Dhaka.

## Models (only these four)

Linear Regression · SVM (RBF) · Random Forest · Decision Tree

**Monthly model** (user input, trained on monthly totals 2021–2024, tested on 2025):

| Model | RMSE (GWh) | MAPE (%) | R² |
| --- | --- | --- | --- |
| **Random Forest (best)** | **269.0** | **5.04** | **0.562** |
| SVM | 272.8 | 6.75 | 0.550 |
| Decision Tree | 285.0 | 5.60 | 0.509 |
| Linear Regression | 347.5 | 7.79 | 0.270 |

**Daily model** (system-level forecast + anomaly days, trained on 2021–2024, tested on 2025):

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | R² |
| --- | --- | --- | --- | --- |
| **SVM (best)** | **365.3** | **286.1** | **5.94** | **0.718** |
| Linear Regression | 368.2 | 285.2 | 6.13 | 0.714 |
| Random Forest | 415.7 | 310.5 | 6.48 | 0.635 |
| Decision Tree | 605.9 | 412.6 | 8.67 | 0.224 |

System results: 4 anomaly days detected in the 2025 test year (residual z-score, \|z\| > 3),
including the Eid-ul-Adha demand swing in June.

## Dataset

[Dhaka 2021–25 Electricity & Temperature data](https://www.kaggle.com/datasets/imuanzumanakhi/dhaka-2021-25-electricity-and-temperature-data)
— 1,759 days of daily electricity demand (MW) and temperature for Dhaka, Bangladesh (Jan 2021 – Oct 2025).

## Training notebook (Kaggle)

The full pipeline runs on Kaggle: [ai-smart-energy-system-training-pipeline](https://www.kaggle.com/code/tamim6/ai-smart-energy-system-training-pipeline).
It loads the dataset, builds simple features, trains the four models, compares them,
forecasts next month, computes the BDT bill, flags anomalies, and trains the monthly
user-input model used by the web app. Outputs are saved to `web/model/`.

## Repository structure

```
├── data/                     # Dhaka 2021–2025 dataset (xlsx)
├── notebooks/                # Kaggle notebook (executed) + kernel metadata
├── web/                      # single-screen Flask app (deployed on Render)
│   ├── app.py                # API: predict, bill, meta, models, health
│   ├── templates/index.html  # the one-screen, no-scroll dashboard
│   └── model/                # artifacts produced by the notebook
└── render.yaml               # Render deploy config
```

## Run locally

```bash
# 1. train (or just use the artifacts already in web/model/)
cd notebooks && jupyter nbconvert --to notebook --execute training_pipeline.ipynb

# 2. serve the web app
cd web && pip install -r requirements.txt && python app.py
```

## Team

Tamim · Emon · Montasir · Muntaha · Shorov — Dhaka International University, Department of CSE.

# AI-Based Smart Electricity Consumption Prediction & Anomaly Detection System

Final-year project — Department of CSE, Dhaka International University.

Predicts **next month's electricity usage** for Dhaka, estimates the **household bill in BDT**
with the official Bangladesh tariff, and **detects anomaly days** — all in one single-page web app.

**Live app:** https://ai-smart-energy-system-w4tx.onrender.com

## What it does

| Feature | Details |
| --- | --- |
| Next-month prediction | Day-by-day recursive forecast for the upcoming month (November 2025) |
| Bill in BDT | Official BERC 2026 residential tariff (LT-A): lifeline + slabs + ৳42/kW demand charge + 5% VAT |
| Anomaly detection | Residual z-score on test days, flagged when \|z\| > 3 |
| Model comparison | Linear Regression vs SVM vs Random Forest vs Decision Tree |

## Dataset

[Dhaka 2021–25 Electricity & Temperature data](https://www.kaggle.com/datasets/imuanzumanakhi/dhaka-2021-25-electricity-and-temperature-data)
— 1,759 days of daily electricity demand (MW) and temperature for Dhaka, Bangladesh (Jan 2021 – Oct 2025).

## Results

Trained on 2021–2024, tested on 2025 (out-of-sample):

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | R² |
| --- | --- | --- | --- | --- |
| **SVM (best)** | **365.3** | **286.1** | **5.94** | **0.718** |
| Linear Regression | 368.2 | 285.2 | 6.13 | 0.714 |
| Random Forest | 415.7 | 310.5 | 6.48 | 0.635 |
| Decision Tree | 605.9 | 412.6 | 8.67 | 0.224 |

November 2025 forecast: **3,092 GWh** total, **4,295 MW** average load,
peak **4,646 MW** on 1 November (−1.8% vs November 2024). Four anomaly days detected in 2025,
including the Eid-ul-Adha demand swing in June.

## Training notebook (Kaggle)

The full pipeline runs on Kaggle: [ai-smart-energy-system-training-pipeline](https://www.kaggle.com/code/tamim6/ai-smart-energy-system-training-pipeline).
It loads the dataset, builds simple lag/calendar/temperature features, trains the four models,
picks the best, forecasts next month, computes the BDT bill table and flags anomalies.
Outputs are saved to `web/model/` for the web app.

## Repository structure

```
├── data/                  # Dhaka 2021–2025 dataset (xlsx)
├── notebooks/             # Kaggle notebook + kernel metadata
├── web/                   # single-page Flask app (deployed on Render)
│   ├── app.py             # Flask API + bill calculator
│   ├── templates/index.html  # the single dashboard page
│   └── model/             # artifacts produced by the notebook
└── render.yaml            # Render deploy config
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

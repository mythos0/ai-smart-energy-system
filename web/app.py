"""
AI-Based Smart Electricity Consumption Prediction & Anomaly Detection System
Flask web application (Render-ready).

Serves:
  - Interactive dashboard (Chart.js): day-ahead forecast backtest, anomaly view,
    model leaderboard, API documentation.
  - REST API:
      GET  /api/health
      GET  /api/meta
      GET  /api/models
      GET  /api/history?start=&end=
      GET  /api/forecast?date=YYYY-MM-DD        (backtest mode)
      POST /api/forecast {"values": [...]}      (live mode, >=192 hourly kW)
      GET  /api/anomaly?start=&end=
      GET  /api/anomaly/casestudy

Model: XGBoost day-ahead forecaster (best of 8 benchmarked models),
anomaly detection via standardised forecast residuals (|z| > 3).
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "model")

app = Flask(__name__)

# ----------------------------------------------------------------- load artifacts
model = xgb.XGBRegressor()
model.load_model(os.path.join(ART, "xgboost.json"))

hourly = pd.read_csv(os.path.join(ART, "hourly_full.csv"), index_col=0, parse_dates=True)
y = hourly["global_active_power"].astype(float)
y.index.name = "timestamp"

preds = pd.read_csv(os.path.join(ART, "test_predictions.csv"), index_col=0, parse_dates=True)
zscore = pd.read_csv(os.path.join(ART, "anomaly_residual_scores.csv"),
                     index_col="timestamp", parse_dates=True)["z"]

with open(os.path.join(ART, "forecast_metrics.json")) as f:
    FMETA = json.load(f)
with open(os.path.join(ART, "anomaly_metrics.json")) as f:
    AMETA = json.load(f)
with open(os.path.join(ART, "anomaly_case_study.json")) as f:
    CASESTUDY = json.load(f)

SIGMA = float(FMETA.get("val_residual_sigma", 0.6325))  # std of XGBoost val residuals

TEST_START = preds.index.min()
TEST_END = preds.index.max()
FEATURES = ["hour", "dayofweek", "month", "is_weekend", "dayofyear",
            "lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_mean_168", "roll_std_24"]

NICE = {"seasonal_naive_24h": "Seasonal naive (24 h)", "seasonal_naive_168h": "Seasonal naive (168 h)",
        "ridge": "Ridge regression", "holt_winters": "Holt-Winters ETS",
        "random_forest": "Random Forest", "xgboost": "XGBoost",
        "lightgbm": "LightGBM", "lstm": "LSTM"}
ADVICE = {"residual_zscore": "Residual z-score (XGBoost)", "isolation_forest": "Isolation Forest",
          "one_class_svm": "One-Class SVM", "lstm_autoencoder": "LSTM autoencoder"}


# ----------------------------------------------------------------- helpers
def build_feature_row(target_ts, series):
    """Features for a single target hour from a Series indexed by timestamp."""
    s = series
    lag24 = s.get(target_ts - pd.Timedelta(hours=24))
    lag48 = s.get(target_ts - pd.Timedelta(hours=48))
    lag168 = s.get(target_ts - pd.Timedelta(hours=168))
    win24 = s[(target_ts - pd.Timedelta(hours=47)):(target_ts - pd.Timedelta(hours=24))]
    win168 = s[(target_ts - pd.Timedelta(hours=191)):(target_ts - pd.Timedelta(hours=24))]
    if any(v is None or pd.isna(v) for v in (lag24, lag48, lag168)):
        return None
    if len(win24) < 24 or len(win168) < 168:
        return None
    return {
        "hour": target_ts.hour, "dayofweek": target_ts.dayofweek, "month": target_ts.month,
        "is_weekend": int(target_ts.dayofweek >= 5), "dayofyear": target_ts.dayofyear,
        "lag_24": float(lag24), "lag_48": float(lag48), "lag_168": float(lag168),
        "roll_mean_24": float(win24.mean()), "roll_mean_168": float(win168.mean()),
        "roll_std_24": float(win24.std()),
    }


def live_forecast(values):
    """values: >=192 most-recent-first hourly kW readings (oldest -> newest)."""
    if len(values) < 192 + 24:
        return None, f"need at least 216 hourly values, got {len(values)}"
    try:
        vals = [float(v) for v in values]
    except (TypeError, ValueError):
        return None, "all values must be numeric"
    if any(v < 0 or v > 20 for v in vals):
        return None, "values out of plausible range (0-20 kW)"
    idx = pd.date_range(end=pd.Timestamp.utcnow().floor("h"), periods=len(vals), freq="h")
    s = pd.Series(vals, index=idx)
    out = []
    for k in range(1, 25):
        ts = idx[-1] + pd.Timedelta(hours=k)
        row = build_feature_row(ts, s)
        if row is None:
            return None, "could not build features from provided history"
        p = float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])
        out.append({"timestamp": ts.isoformat(), "power_kw": round(max(p, 0.0), 4)})
        s.loc[ts] = p  # feed prediction back for recursive lags beyond +24h window
    return out, None


# ----------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html",
                           test_start=TEST_START.strftime("%Y-%m-%d"),
                           test_end=TEST_END.strftime("%Y-%m-%d"),
                           default_day=(TEST_END - pd.Timedelta(days=7)).strftime("%Y-%m-%d"))


# ----------------------------------------------------------------- API
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "xgboost", "horizon": "24h"})


@app.route("/api/meta")
def meta():
    return jsonify({
        "dataset": "UCI Individual Household Electric Power Consumption (2006-2010)",
        "hourly_rows": int(len(y)),
        "test_range": [str(TEST_START), str(TEST_END)],
        "target": "global_active_power (kW), hourly mean",
        "task": "24-hour-ahead day-ahead forecasting + residual-based anomaly detection",
        "forecast_metrics": FMETA.get("summary_table", []),
        "anomaly_metrics": {ADVICE.get(k, k): v for k, v in AMETA["results"].items()},
        "best_model": "xgboost",
    })


@app.route("/api/models")
def models():
    rows = []
    for r in sorted(FMETA.get("summary_table", []), key=lambda d: d["rmse"]):
        rows.append({"model": NICE.get(r["model"], r["model"]), **{k: round(v, 4) if isinstance(v, float) else v
                                                                   for k, v in r.items()}})
    det = [{"model": ADVICE.get(k, k), **{m: round(v, 4) for m, v in r.items()
                                          if isinstance(v, (int, float))}}
           for k, r in AMETA["results"].items() if k != "_protocol"]
    return jsonify({"forecasting": rows, "anomaly_detection": det})


@app.route("/api/history")
def history():
    start = pd.Timestamp(request.args.get("start"))
    end = pd.Timestamp(request.args.get("end"))
    seg = y.loc[start:end]
    return jsonify({"series": [{"timestamp": str(ts), "power_kw": round(float(v), 4)}
                               for ts, v in seg.items()]})


@app.route("/api/forecast")
def forecast():
    date = request.args.get("date")
    try:
        day = pd.Timestamp(date)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid date"}), 400
    if day < TEST_START or day > TEST_END:
        return jsonify({"error": f"date must be within test range "
                                 f"{TEST_START.date()} .. {TEST_END.date()}"}), 400
    hours = pd.date_range(day, periods=24, freq="h")
    rows = []
    for ts in hours:
        p = preds["xgboost"].get(ts)
        a = y.get(ts)
        rows.append({"timestamp": str(ts),
                     "forecast_kw": None if p is None or pd.isna(p) else round(float(p), 4),
                     "actual_kw": None if a is None or pd.isna(a) else round(float(a), 4)})
    return jsonify({"mode": "backtest", "date": str(day.date()), "horizon_h": 24,
                    "sigma_kw": round(SIGMA, 4), "points": rows})


@app.route("/api/forecast", methods=["POST"])
def forecast_live():
    payload = request.get_json(silent=True) or {}
    values = payload.get("values")
    if not isinstance(values, list):
        return jsonify({"error": "JSON body must contain 'values': [hourly kW readings]"}), 400
    out, err = live_forecast(values)
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"mode": "live", "horizon_h": 24, "points": out})


@app.route("/api/anomaly")
def anomaly():
    start = pd.Timestamp(request.args.get("start"))
    end = pd.Timestamp(request.args.get("end"))
    zseg = zscore.loc[start:end]
    return jsonify({"sigma": "validation residual std", "points": [
        {"timestamp": str(ts), "z": round(float(v), 3), "is_anomaly": bool(v > 3)}
        for ts, v in zseg.items()]})


@app.route("/api/anomaly/casestudy")
def casestudy():
    return jsonify({"events": CASESTUDY})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

"""
AI-Based Smart Electricity Consumption Prediction & Anomaly Detection System
Single-screen Flask web app (Render-ready). No scrolling needed.

One page:
  GET  /            single-screen dashboard (fits mobile + desktop, no scroll)

API:
  GET  /api/health    service status
  GET  /api/meta      dataset + system stats
  GET  /api/models    model comparison tables (daily + monthly)
  POST /api/predict   {m1, m2, m3, m4, kw} -> next-month usage, bill in BDT, anomaly check
  POST /api/bill      {units, kw}          -> standalone bill calculator

User input: the last 4 monthly bills (kWh). The monthly model is scale-free
(ratios to the 3-month average), so the same model works for any consumer.
Bill: Bangladesh residential tariff (BERC 2026, category LT-A).
Anomaly detection: residual z-score (|z| > 3) on the latest month.
"""
import json
import os
from calendar import month_abbr, month_name
from datetime import date
from math import cos, pi, sin

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "model")

app = Flask(__name__)

# ----------------------------------------------------------------- load artifacts
daily = pd.read_csv(os.path.join(ART, "daily.csv"), parse_dates=["date"])
fcast = json.load(open(os.path.join(ART, "next_month.json")))
models_meta = json.load(open(os.path.join(ART, "models.json")))          # daily models
monthly_meta = json.load(open(os.path.join(ART, "monthly_meta.json")))   # monthly models
monthly_models = joblib.load(os.path.join(ART, "monthly_models.joblib"))
test_preds = pd.read_csv(os.path.join(ART, "test_predictions.csv"), parse_dates=["date"])
monthly_preds = pd.read_csv(os.path.join(ART, "monthly_test_predictions.csv"))

FEATS = monthly_meta["features"]        # ["r1", "r2", "r3", "msin", "mcos"]
RSTD = monthly_meta["resid_std"]        # out-of-sample residual spread (ratio scale)
BEST = monthly_meta["best_model"]
ANOMALY_DAYS = int(test_preds["is_anomaly"].sum())

# ----------------------------------------------------------------- bill (BERC 2026, residential LT-A)
STEPPED = [(1, 75, 5.26), (76, 200, 8.50), (201, 300, 9.10),
           (301, 400, 9.62), (401, 600, 15.01), (601, None, 17.35)]
LIFELINE_RATE, LIFELINE_MAX = 4.63, 50      # lifeline applies only if monthly use <= 50 units
DEMAND_CHARGE, VAT = 42.0, 0.05             # 42 BDT per kW load, 5% VAT


def bill_bdt(units, kw=1.0):
    """Return the monthly bill breakdown for a household."""
    if units <= LIFELINE_MAX:
        # lifeline: every unit at the cheap rate
        energy = units * LIFELINE_RATE
    else:
        # stepped slabs: each rate applies only to units inside its band
        energy, prev = 0.0, 0
        for lo, hi, rate in STEPPED:
            top = hi if hi is not None else units
            energy += max(0, min(units, top) - lo + 1) * rate
            prev = hi
    demand = DEMAND_CHARGE * kw
    vat = (energy + demand) * VAT
    return {
        "units": round(units, 1), "kw": kw,
        "energy_charge": round(energy, 2),
        "demand_charge": round(demand, 2),
        "vat": round(vat, 2),
        "total": round(energy + demand + vat, 2),
        "per_unit": round((energy + demand + vat) / units, 2) if units else 0,
    }


# ----------------------------------------------------------------- monthly prediction helpers
def feature_row(l1, l2, l3, month_no):
    """Scale-free feature row from the 3 months before the month being predicted."""
    base = (l1 + l2 + l3) / 3
    x = pd.DataFrame([[l1 / base, l2 / base, l3 / base,
                       sin(2 * pi * month_no / 12), cos(2 * pi * month_no / 12)]],
                     columns=FEATS)
    return x, base


def predict_user(m1, m2, m3, m4, kw, today):
    """Predict next-month usage, the bill in BDT, and check the latest month."""
    cur, nxt = today.month, today.month % 12 + 1

    # 1) anomaly check on the latest month (m4): predict it from m1, m2, m3
    x4, base3 = feature_row(m3, m2, m1, cur)
    exp4 = float(monthly_models[BEST].predict(x4)[0]) * base3
    z4 = (m4 - exp4) / (RSTD * base3)
    verdict = "high" if z4 > 3 else ("low" if z4 < -3 else "normal")

    # 2) next-month usage from m2, m3, m4 (all four models, best highlighted)
    x5, base4 = feature_row(m4, m3, m2, nxt)
    per_model = {name: round(float(m.predict(x5)[0]) * base4, 1)
                 for name, m in monthly_models.items()}
    next_kwh = per_model[BEST]

    # 3) predicted bill for the forecast usage
    bill = bill_bdt(next_kwh, kw)

    # month labels for the UI (1st-4th bill + short names for context)
    nxt_y = today.year + (1 if today.month == 12 else 0)

    return {
        "labels": {"m1": "1st", "m2": "2nd", "m3": "3rd", "m4": "4th",
                   "next": month_abbr[nxt]},
        "next_name": f"{month_name[nxt]} {nxt_y}",
        "checked_name": f"{month_name[cur]} {today.year}",
        "checked_short": month_abbr[cur],
        "inputs": {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "kw": kw},
        "next_kwh": next_kwh,
        "per_model": per_model,
        "best_model": BEST,
        "bill": bill,
        "anomaly": {"actual": round(m4, 1), "expected": round(exp4, 1),
                    "gap": round(m4 - exp4, 1),
                    "gap_pct": round((m4 - exp4) / exp4 * 100, 1),
                    "z": round(z4, 2), "verdict": verdict},
    }


# ----------------------------------------------------------------- page
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------- API
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "page": "single-screen dashboard",
                    "models": ["Linear Regression", "SVM", "Random Forest", "Decision Tree"],
                    "best_model": BEST})


@app.route("/api/meta")
def meta():
    daily_best = models_meta["best_model"]
    daily_rmse = min(r["RMSE"] for r in models_meta["results"])
    monthly_mape = [r["MAPE (%)"] for r in monthly_meta["results"]
                    if r["model"] == BEST][0]
    return jsonify({
        "dataset": "Dhaka daily electricity demand & temperature, 2021-2025 (Kaggle)",
        "daily_rows": int(len(daily)),
        "last_day": daily["date"].max().strftime("%Y-%m-%d"),
        "daily_model": daily_best,
        "daily_rmse_mw": daily_rmse,
        "test_period": models_meta["test_period"],
        "anomaly_days_2025": ANOMALY_DAYS,
        "monthly_model": BEST,
        "monthly_mape_pct": monthly_mape,
        "grid_forecast": {"month": fcast["month_name"],
                          "total_gwh": fcast["stats"]["total_gwh"],
                          "yoy_pct": fcast["stats"]["yoy_pct"]},
        "tariff": "Bangladesh residential (BERC 2026, LT-A), 5% VAT included",
    })


@app.route("/api/models")
def api_models():
    return jsonify({"daily_demand_model": models_meta,
                    "monthly_user_model": monthly_meta})


@app.route("/api/evaluation")
def api_evaluation():
    """Test-period series behind the evaluation charts on the dashboard."""
    # monthly: complete months only (the last month of the dataset is partial)
    mp = monthly_preds[monthly_preds["days"] >= 28]
    monthly_series = [{"month": r["month"],
                       "label": month_abbr[int(r["month"].split("-")[1])],
                       "actual": float(r["actual_gwh"]),
                       "predicted": float(r["predicted_gwh"])}
                      for _, r in mp.iterrows()]

    # daily: 2025 test days, actual vs model expectation
    daily_series = [{"date": d.strftime("%Y-%m-%d"), "actual": float(a),
                     "expected": float(p), "anomaly": int(i)}
                    for d, a, p, i in zip(test_preds["date"], test_preds["actual_mw"],
                                          test_preds["predicted_mw"], test_preds["is_anomaly"])]

    ev = test_preds[test_preds["is_anomaly"] == 1]
    ev = ev.reindex(ev["z"].abs().sort_values(ascending=False).index)
    events = [{"date": d.strftime("%Y-%m-%d"), "actual": float(a), "expected": float(p),
               "z": float(z), "type": "HIGH" if z > 0 else "LOW"}
              for d, a, p, z in zip(ev["date"], ev["actual_mw"], ev["predicted_mw"], ev["z"])]

    return jsonify({
        "monthly": {"model": monthly_meta["best_model"], "series": monthly_series},
        "daily": {"model": models_meta["best_model"], "series": daily_series,
                  "events": events},
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(silent=True) or {}
    try:
        m1, m2, m3, m4 = (float(body.get(k, 0)) for k in ("m1", "m2", "m3", "m4"))
        kw = float(body.get("kw", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "m1, m2, m3, m4 and kw must be numbers"}), 400
    if any(not (1 <= v <= 50000) for v in (m1, m2, m3, m4)):
        return jsonify({"error": "each month must be 1 - 50000 kWh"}), 400
    if not (1 <= kw <= 50):
        return jsonify({"error": "kw must be 1 - 50"}), 400
    return jsonify(predict_user(m1, m2, m3, m4, kw, date.today()))


@app.route("/api/bill", methods=["POST"])
def api_bill():
    body = request.get_json(silent=True) or {}
    try:
        units = float(body.get("units", 250))
        kw = float(body.get("kw", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "units and kw must be numbers"}), 400
    if not (1 <= units <= 50000) or not (1 <= kw <= 50):
        return jsonify({"error": "units must be 1-50000 and kw 1-50"}), 400
    return jsonify(bill_bdt(units, kw))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

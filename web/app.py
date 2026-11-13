"""
AI-Based Smart Electricity Consumption Prediction & Anomaly Detection System
Single-page Flask web application (Render-ready).

One page:
  GET  /            dashboard (next-month forecast + BDT bill estimator + anomalies + models)

API:
  GET  /api/health
  GET  /api/meta
  GET  /api/daily           full daily demand series
  GET  /api/forecast        next-month prediction + stats
  GET  /api/anomalies       detected anomaly events
  GET  /api/models          model comparison table
  POST /api/bill            {"units": 250, "kw": 1} -> bill in BDT

Prediction: best of 4 models (Linear Regression, SVM, Random Forest,
Decision Tree) trained on Dhaka daily electricity demand 2021-2025.
Bill: Bangladesh residential tariff (BERC 2026, category LT-A).
Anomaly detection: residual z-score (|z| > 3).
"""
import json
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "model")

app = Flask(__name__)

# ----------------------------------------------------------------- load artifacts
daily = pd.read_csv(os.path.join(ART, "daily.csv"), parse_dates=["date"])
fcast = json.load(open(os.path.join(ART, "next_month.json")))
models_meta = json.load(open(os.path.join(ART, "models.json")))
test_preds = pd.read_csv(os.path.join(ART, "test_predictions.csv"), parse_dates=["date"])

LAST_DAY = daily["date"].max().strftime("%Y-%m-%d")

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
        rows = [{"slab": "Lifeline (1-50)", "rate": LIFELINE_RATE,
                 "units": units, "charge": round(energy, 2)}]
        rows += [{"slab": slab_name(lo, hi), "rate": r, "units": 0, "charge": 0.0}
                 for lo, hi, r in STEPPED]
    else:
        # stepped slabs: each rate applies only to units inside its band
        energy, rows = 0.0, [{"slab": "Lifeline (1-50)", "rate": LIFELINE_RATE,
                              "units": 0, "charge": 0.0}]
        for lo, hi, rate in STEPPED:
            top = hi if hi is not None else units
            n = max(0, min(units, top) - lo + 1)
            charge = n * rate
            energy += charge
            rows.append({"slab": slab_name(lo, hi), "rate": rate,
                         "units": n, "charge": round(charge, 2)})
    demand = DEMAND_CHARGE * kw
    vat = (energy + demand) * VAT
    return {
        "units": units, "kw": kw,
        "energy_charge": round(energy, 2),
        "demand_charge": round(demand, 2),
        "vat": round(vat, 2),
        "total": round(energy + demand + vat, 2),
        "per_unit": round((energy + demand + vat) / units, 2) if units else 0,
        "slab_detail": rows,
    }


def slab_name(lo, hi):
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


# ----------------------------------------------------------------- page
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------- API
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "page": "single-page dashboard",
                    "models": ["Linear Regression", "SVM", "Random Forest", "Decision Tree"],
                    "best_model": models_meta["best_model"]})


@app.route("/api/meta")
def meta():
    return jsonify({
        "dataset": "Dhaka daily electricity demand & temperature, 2021-2025 (Kaggle)",
        "daily_rows": int(len(daily)),
        "last_day": LAST_DAY,
        "target": "daily demand (MW)",
        "best_model": models_meta["best_model"],
        "test_period": models_meta["test_period"],
        "next_month": fcast["month_name"],
        "tariff": "Bangladesh residential (BERC 2026, LT-A), 5% VAT included",
    })


@app.route("/api/daily")
def api_daily():
    seg = daily[["date", "demand_mw", "temp_c"]]
    return jsonify({"series": [{"date": d.strftime("%Y-%m-%d"),
                                "demand_mw": float(v), "temp_c": float(t)}
                               for d, v, t in seg.itertuples(index=False)]})


@app.route("/api/forecast")
def api_forecast():
    out = dict(fcast)
    tail = daily[["date", "demand_mw"]].tail(90)
    out["history_tail"] = [{"date": d.strftime("%Y-%m-%d"), "demand_mw": float(v)}
                           for d, v in tail.itertuples(index=False)]
    return jsonify(out)


@app.route("/api/anomalies")
def api_anomalies():
    seg = test_preds[test_preds["is_anomaly"] == 1].copy()
    seg = seg.reindex(seg["z"].abs().sort_values(ascending=False).index)
    series = [{"date": d.strftime("%Y-%m-%d"), "actual_mw": float(a),
               "expected_mw": float(p), "z": float(z), "is_anomaly": int(i)}
              for d, a, p, z, i in test_preds[["date", "actual_mw", "predicted_mw", "z",
                                               "is_anomaly"]].itertuples(index=False)]
    return jsonify({"method": "residual z-score, |z| > 3",
                    "events": [
        {"date": d.strftime("%Y-%m-%d"), "actual_mw": float(a), "expected_mw": float(p),
         "gap_mw": round(a - p, 1), "z": float(z),
         "type": "unusual HIGH" if z > 0 else "unusual LOW"}
        for d, a, p, z in seg[["date", "actual_mw", "predicted_mw", "z"]].itertuples(index=False)],
        "series": series})


@app.route("/api/models")
def api_models():
    return jsonify(models_meta)


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

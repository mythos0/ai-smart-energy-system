"""Build the Kaggle training notebook (.ipynb) from cell sources."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# AI-Based Smart Electricity Consumption Prediction & Anomaly Detection

Reproducible training notebook for the research paper
**"AI-Based Smart Electricity Consumption Prediction and Anomaly Detection: A Benchmark Study and Deployable Web System"**.

**Pipeline:** UCI minute-level data -> cleaning -> hourly aggregation -> strict day-ahead features -> 8 forecasting models -> 4 anomaly detectors -> results tables.

**Protocol notes**
- Task: *day-ahead* forecasting — every feature for target hour t is at least 24 h old.
- Chronological splits: train (to 2009-06-30), validation (2009-07-01..2010-03-31), test (2010-04-01..2010-11-26).
- Anomaly benchmark: 2% of test hours injected (spikes / level shifts / dropouts), lag features recomputed from the contaminated series.
- Runtime: ~15 min on a CPU-only kernel.""")

code("""import json, time, warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(2)
print("torch", torch.__version__, "| xgboost", xgb.__version__, "| lightgbm", lgb.__version__)""")

md("## 1. Load and clean the raw data")

code("""RAW = "/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"

df = pd.read_csv(RAW, sep=";", na_values="?",
                 parse_dates={"timestamp": ["Date", "Time"]}, dayfirst=True,
                 low_memory=False)
print(f"raw rows: {len(df):,}")
n_missing = int(df.isna().any(axis=1).sum())
print(f"rows with missing values: {n_missing:,} ({n_missing/len(df)*100:.2f}%)")

df = df.set_index("timestamp").sort_index()
df = df.interpolate(method="time", limit=60).dropna()
print(f"rows after cleaning: {len(df):,}")

# hourly aggregation of global active power (kW)
y = df["Global_active_power"].resample("h").mean().dropna()
print(f"hourly rows: {len(y):,}  range: {y.index.min()} .. {y.index.max()}")""")

md("## 2. Strict day-ahead feature engineering\nEvery feature for target hour $t$ is computed from data **older than $t-23$ h** (lags 24/48/168 h and rolling statistics shifted by 24 h).")

code("""TRAIN_END, VAL_END = "2009-06-30 23:00", "2010-03-31 23:00"

def build_features(series):
    X = pd.DataFrame(index=series.index)
    X["gap"] = series
    X["hour"] = series.index.hour
    X["dayofweek"] = series.index.dayofweek
    X["month"] = series.index.month
    X["is_weekend"] = (series.index.dayofweek >= 5).astype(int)
    X["dayofyear"] = series.index.dayofyear
    X["lag_24"] = series.shift(24)
    X["lag_48"] = series.shift(48)
    X["lag_168"] = series.shift(168)
    X["roll_mean_24"] = series.shift(24).rolling(24).mean()
    X["roll_mean_168"] = series.shift(24).rolling(168).mean()
    X["roll_std_24"] = series.shift(24).rolling(24).std()
    return X.dropna()

FEATURES = ["hour", "dayofweek", "month", "is_weekend", "dayofyear",
            "lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_mean_168", "roll_std_24"]

X = build_features(y)
tr, va, te = X[X.index <= TRAIN_END], X.query(f"'{TRAIN_END}' < index <= '{VAL_END}'"), X[X.index > VAL_END]
y_tr, y_va, y_te = tr["gap"], va["gap"], te["gap"]
print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")

def metrics(yt, yp):
    return {"rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "mape": float(np.mean(np.abs((yt - yp) / yt)) * 100),
            "r2": float(r2_score(yt, yp))}

results = {}""")

md("## 3. Baselines, Ridge, and Holt-Winters")

code("""# seasonal naive
p24 = y.shift(24).loc[te.index]; p168 = y.shift(168).loc[te.index]
results["seasonal_naive_24h"] = metrics(y_te, p24)
results["seasonal_naive_168h"] = metrics(y_te, p168)

# ridge with cyclical encoding
def cyc(d):
    Z = pd.DataFrame(index=d.index)
    a = lambda p, q: 2 * np.pi * d[p] / q
    Z["sin_h"], Z["cos_h"] = np.sin(a("hour", 24)), np.cos(a("hour", 24))
    Z["sin_d"], Z["cos_d"] = np.sin(a("dayofweek", 7)), np.cos(a("dayofweek", 7))
    Z["sin_m"], Z["cos_m"] = np.sin(a("month", 12)), np.cos(a("month", 12))
    for c in ["is_weekend", "lag_24", "lag_48", "lag_168",
              "roll_mean_24", "roll_mean_168", "roll_std_24"]:
        Z[c] = d[c]
    return Z
Ztr, Zva, Zte = cyc(tr), cyc(va), cyc(te)
best = min([0.1, 1, 10, 100], key=lambda a: mean_squared_error(y_va, Ridge(alpha=a).fit(Ztr, y_tr).predict(Zva)))
ridge = Ridge(alpha=best).fit(Ztr, y_tr)
results["ridge"] = metrics(y_te, ridge.predict(Zte))
print("ridge alpha:", best)

# Holt-Winters: daily refits on trailing 8-week window (timestamps assigned explicitly)
def hw_forecast(series, target_index, seasonal="add", window=8 * 168):
    out = {}
    for d in sorted(set(target_index.normalize())):
        hist = series[series.index < d].iloc[-window:]
        try:
            m = ExponentialSmoothing(hist.values, trend="add", damped_trend=True,
                                     seasonal=seasonal, seasonal_periods=24,
                                     initialization_method="estimated").fit(optimized=True)
            for ts, v in zip(pd.date_range(d, periods=24, freq="h"), np.asarray(m.forecast(24))):
                if ts in target_index:
                    out[ts] = float(v)
        except Exception:
            pass
    return pd.Series(out)

t0 = time.time()
best_hw = min(["add", "mul"], key=lambda s: mean_squared_error(
    y_va, hw_forecast(y.loc[:VAL_END], va.index, s).reindex(va.index).dropna()))
p_hw = hw_forecast(y, te.index, best_hw).reindex(te.index).ffill()
results["holt_winters"] = metrics(y_te, p_hw)
print(f"HW seasonal={best_hw} ({time.time()-t0:.0f}s)")""")

md("## 4. Tree ensembles (Random Forest, XGBoost, LightGBM)")

code("""Ftr, Fva, Fte = tr[FEATURES], va[FEATURES], te[FEATURES]

# Random Forest
best = min([(d, l) for d in [18, None] for l in [1, 4]],
           key=lambda dl: mean_squared_error(y_va, RandomForestRegressor(
               n_estimators=300, max_depth=dl[0], min_samples_leaf=dl[1],
               n_jobs=-1, random_state=SEED).fit(Ftr, y_tr).predict(Fva)))
rf = RandomForestRegressor(n_estimators=500, max_depth=best[0], min_samples_leaf=best[1],
                           n_jobs=-1, random_state=SEED).fit(Ftr, y_tr)
results["random_forest"] = metrics(y_te, rf.predict(Fte))
print("RF best (depth, leaf):", best)

# XGBoost
best = min([(n, d, lr) for n, d, lr in [(400, 4, .06), (800, 4, .03), (600, 5, .05)]],
           key=lambda p: mean_squared_error(y_va, xgb.XGBRegressor(
               n_estimators=p[0], max_depth=p[1], learning_rate=p[2], subsample=.9,
               colsample_bytree=.9, reg_lambda=1.0, tree_method="hist",
               random_state=SEED).fit(Ftr, y_tr, verbose=False).predict(Fva)))
xg = xgb.XGBRegressor(n_estimators=best[0], max_depth=best[1], learning_rate=best[2],
                      subsample=.9, colsample_bytree=.9, reg_lambda=1.0,
                      tree_method="hist", random_state=SEED).fit(Ftr, y_tr)
results["xgboost"] = metrics(y_te, xg.predict(Fte))
xg.save_model("xgboost.json")
print("XGB best:", best)

# LightGBM
best = min([(n, l, lr) for n, l, lr in [(400, 31, .06), (800, 31, .03), (600, 47, .05)]],
           key=lambda p: mean_squared_error(y_va, lgb.LGBMRegressor(
               n_estimators=p[0], num_leaves=p[1], learning_rate=p[2], subsample=.9,
               colsample_bytree=.9, reg_lambda=1.0, random_state=SEED,
               verbose=-1).fit(Ftr, y_tr).predict(Fva)))
lg = lgb.LGBMRegressor(n_estimators=best[0], num_leaves=best[1], learning_rate=best[2],
                       subsample=.9, colsample_bytree=.9, reg_lambda=1.0,
                       random_state=SEED, verbose=-1).fit(Ftr, y_tr)
results["lightgbm"] = metrics(y_te, lg.predict(Fte))
print("LGBM best:", best)""")

md("## 5. LSTM (direct multi-step: 168 h -> 24 h)\nInput channels: standardised consumption + cyclical hour/day-of-week. Scoring uses output step 24 of the window ending 24 h before the target — the same feature timeliness as the tree models.")

code("""PAST, HORIZON = 168, 24
mu, sd = float(y_tr.mean()), float(y_tr.std())
ya = (y - mu) / sd
M = np.stack([ya.values,
              np.sin(2 * np.pi * y.index.hour / 24),
              np.cos(2 * np.pi * y.index.hour / 24),
              np.sin(2 * np.pi * y.index.dayofweek / 7),
              np.cos(2 * np.pi * y.index.dayofweek / 7)], axis=1)
pos = {ts: i for i, ts in enumerate(y.index)}

def windows(idx):
    Xs, Ys, TS = [], [], []
    for ts in idx:
        i, j = pos[ts], pos[ts] - HORIZON
        if j - PAST + 1 >= 0:
            Xs.append(M[j - PAST + 1: j + 1]); Ys.append(ya.values[j + 1: j + HORIZON + 1]); TS.append(ts)
    return (torch.tensor(np.array(Xs), dtype=torch.float32),
            torch.tensor(np.array(Ys), dtype=torch.float32), TS)

Xtr, Ytr, _ = windows(tr.index)
Xva, Yva, _ = windows(va.index)
Xte, _, TSte = windows(te.index)
print("tensors:", Xtr.shape, Xva.shape, Xte.shape)

class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 64, 2, batch_first=True, dropout=0.15)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 24))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :])

model = LSTMForecaster()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
lossf = nn.SmoothL1Loss()
trl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)
vld = DataLoader(TensorDataset(Xva, Yva), batch_size=256)
best_state, best_vl, wait = None, np.inf, 0
t0 = time.time()
for ep in range(40):
    model.train()
    for xb, yb in trl:
        opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    model.eval(); vl = []
    with torch.no_grad():
        for xb, yb in vld:
            vl.append(lossf(model(xb), yb).item())
    vl = float(np.mean(vl))
    if vl < best_vl:
        best_vl, wait, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
    if wait >= 6:
        break
print(f"LSTM trained {ep+1} epochs in {time.time()-t0:.0f}s (best val {best_vl:.4f})")
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    phat = model(Xte).numpy() * sd + mu
p_lstm = pd.Series(phat[:, -1], index=TSte)
results["lstm"] = metrics(y_te.reindex(p_lstm.index), p_lstm)
torch.save({"state_dict": best_state, "mu": mu, "sd": sd,
            "config": {"hidden": 64, "layers": 2, "in_dim": 5, "horizon": 24, "past": 168}},
           "lstm_forecaster.pt")""")

md("## 6. Forecasting results")

code("""rows = [{"model": k, **v,
          "skill_%": round((1 - v["rmse"] / results["seasonal_naive_24h"]["rmse"]) * 100, 1)}
         for k, v in results.items()]
summary = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
summary.round(4)""")

md("## 7. Anomaly detection benchmark\nInject 2% of test hours (spikes / level shifts / dropouts), recompute lag features from the **contaminated** series, and compare four detectors at the prevalence-matched operating point.")

code("""def inject(series, seed=SEED, rate=0.02):
    r = np.random.default_rng(seed)
    s = series.copy()
    labels = pd.Series(0, index=series.index, dtype=int)
    atype = pd.Series("", index=series.index, dtype=object)
    tp = np.where(series.index > VAL_END)[0]
    budget = int(len(tp) * rate); used = set()
    n_spike = int(budget * .4)
    for p in r.choice(tp, size=n_spike, replace=False):
        s.iloc[p] = min(s.iloc[p] * r.uniform(2.5, 4.0) + 1.5, 8.0)
        labels.iloc[p] = 1; atype.iloc[p] = "spike"; used.add(int(p))
    placed = 0; target = int(budget * .3)
    while placed < target:
        p = int(r.choice(tp)); L = int(r.integers(2, 7))
        block = list(range(p, min(p + L, len(s))))
        if not block or any(b in used for b in block) or p < tp.min(): continue
        f = r.uniform(1.6, 2.5)
        for b in block:
            s.iloc[b] = min(s.iloc[b] * f, 8.0); labels.iloc[b] = 1; atype.iloc[b] = "shift"
        used.update(block); placed += len(block)
    target = budget - n_spike - placed; placed = 0
    while placed < target:
        p = int(r.choice(tp)); L = int(r.integers(4, 13))
        block = list(range(p, min(p + L, len(s))))
        if not block or any(b in used for b in block) or p < tp.min(): continue
        for b in block:
            s.iloc[b] = r.uniform(.02, .10); labels.iloc[b] = 1; atype.iloc[b] = "dropout"
        used.update(block); placed += len(block)
    return s, labels, atype

y_cor, y_lab, y_typ = inject(y)
X_cor = build_features(y_cor)
te_cor = X_cor[X_cor.index > VAL_END]
lab = y_lab.reindex(te_cor.index).values.astype(int)
typ = y_typ.reindex(te_cor.index).values
rate = lab.mean(); Q = 1 - rate
print(f"injected: {lab.sum()} hours ({rate*100:.2f}%) | "
      f"spike={(typ=='spike').sum()} shift={(typ=='shift').sum()} dropout={(typ=='dropout').sum()}")

def prf(score):
    pred = (score >= np.quantile(score, Q)).astype(int)
    out = {"precision": precision_score(lab, pred, zero_division=0),
           "recall": recall_score(lab, pred, zero_division=0),
           "f1": f1_score(lab, pred, zero_division=0),
           "roc_auc": roc_auc_score(lab, score), "pr_auc": average_precision_score(lab, score)}
    for t in ["spike", "shift", "dropout"]:
        m = typ == t
        out[f"recall_{t}"] = float((pred[m] == 1).mean()) if m.sum() else None
    return out

anom = {}

# A) residual z-score with XGBoost
sigma = float(np.std(y_va.values - xg.predict(Fva)))
anom["residual_zscore"] = prf(np.abs(te_cor["gap"].values - xg.predict(te_cor[FEATURES])) / sigma)

# B) Isolation Forest
IF_F = ["gap", "lag_24", "lag_168", "roll_mean_24", "roll_std_24", "hour", "dayofweek"]
sc = StandardScaler().fit(tr[IF_F])
iso = IsolationForest(n_estimators=300, contamination=float(rate), random_state=SEED,
                      n_jobs=-1).fit(sc.transform(tr[IF_F]))
anom["isolation_forest"] = prf(-iso.score_samples(sc.transform(te_cor[IF_F])))

# C) One-Class SVM
oc = OneClassSVM(kernel="rbf", nu=float(rate), gamma="scale").fit(
    sc.transform(tr[IF_F].sample(n=min(12000, len(tr)), random_state=SEED)))
anom["one_class_svm"] = prf(-oc.decision_function(sc.transform(te_cor[IF_F])).ravel())

# D) LSTM autoencoder (24h windows, per-hour localised error)
class LSTMAE(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.enc = nn.LSTM(1, hidden, 1, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, 1, batch_first=True)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        o, (hn, _) = self.enc(x)
        z = hn[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        d, _ = self.dec(z)
        return self.out(d)

W = 24
zc = ((y_cor - mu) / sd).values
Xs, P0 = [], []
for ts in te_cor.index:
    p = pos[ts]
    if p - W + 1 >= 0:
        Xs.append(zc[p - W + 1: p + 1]); P0.append(p - W + 1)
Xt = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1)
ae = LSTMAE()
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
trl = DataLoader(TensorDataset(Xt[:len(tr)], Xt[:len(tr)]), batch_size=128, shuffle=True)
for ep in range(15):
    ae.train()
    for xb, _ in trl:
        opt.zero_grad(); nn.MSELoss()(ae(xb), xb).backward(); opt.step()
ae.eval()
with torch.no_grad():
    rec = ae(Xt).squeeze(-1).numpy()
err = (rec - Xt.squeeze(-1).numpy()) ** 2
he, hc = np.zeros(len(zc)), np.zeros(len(zc))
for wi, p0 in enumerate(P0):
    for k in range(W):
        he[p0 + k] += err[wi, k]; hc[p0 + k] += 1
hour_score = np.where(hc > 0, he / np.maximum(hc, 1), 0)
anom["lstm_autoencoder"] = prf(pd.Series(hour_score, index=y.index).reindex(te_cor.index).fillna(0).values)

pd.DataFrame(anom).T.round(4)""")

md("## 8. Summary\nThe tables above reproduce the paper's Tables 3 and 4. Expected outcomes: **XGBoost** leads forecasting (RMSE ~0.588 kW, ~22% skill over naive-24h); the **residual z-score** leads anomaly detection (ROC-AUC ~0.885), with high spike recall but near-zero dropout recall across all detectors. Trained artefacts (`xgboost.json`, `lstm_forecaster.pt`) are saved to the working directory for the deployed web system.")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.11"}}
nbf.write(nb, "/home/z/my-project/notebooks/training_pipeline.ipynb")
print("notebook written:", len(cells), "cells")

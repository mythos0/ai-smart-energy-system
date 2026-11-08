"""
Forecasting model training & selection.  (resumable: skips models already in results file)

Task : 24-hour-ahead (day-ahead) hourly forecasting of Global Active Power (kW).
       Features for predicting hour t use ONLY information older than t-23h.
Models: seasonal naive 24h/168h, Ridge, Holt-Winters ETS, Random Forest,
        XGBoost, LightGBM, LSTM (direct multi-step, window 168h -> 24h).
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

DATA = "/home/z/my-project/data"
RES = "/home/z/my-project/results"
MOD = "/home/z/my-project/models"
os.makedirs(RES, exist_ok=True)
os.makedirs(MOD, exist_ok=True)
CKPT = f"{RES}/forecast_metrics.json"

# ----------------------------------------------------------------- data
h = pd.read_csv(f"{DATA}/hourly_full.csv", index_col=0, parse_dates=True)
y = h["global_active_power"]

X = pd.DataFrame(index=y.index)
X["gap"] = y
X["hour"] = y.index.hour
X["dayofweek"] = y.index.dayofweek
X["month"] = y.index.month
X["is_weekend"] = (y.index.dayofweek >= 5).astype(int)
X["dayofyear"] = y.index.dayofyear
X["lag_24"] = y.shift(24)
X["lag_48"] = y.shift(48)
X["lag_168"] = y.shift(168)
X["roll_mean_24"] = y.shift(24).rolling(24).mean()
X["roll_mean_168"] = y.shift(24).rolling(168).mean()
X["roll_std_24"] = y.shift(24).rolling(24).std()
X = X.dropna()

TRAIN_END, VAL_END = "2009-06-30 23:00", "2010-03-31 23:00"
tr = X[X.index <= TRAIN_END]
va = X[(X.index > TRAIN_END) & (X.index <= VAL_END)]
te = X[X.index > VAL_END]
y_tr, y_va, y_te = tr["gap"], va["gap"], te["gap"]
FEATURES = ["hour", "dayofweek", "month", "is_weekend", "dayofyear",
            "lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_mean_168", "roll_std_24"]

def metrics(y_true, y_pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            "r2": float(r2_score(y_true, y_pred))}

# ----------------------------------------------------------------- resume state
results, preds_test = {}, {}
if os.path.exists(CKPT):
    d = json.load(open(CKPT))
    results = d.get("results", {})
    print(f"RESUME: already computed -> {list(results)}")
    if os.path.exists(f"{RES}/test_predictions.csv"):
        P = pd.read_csv(f"{RES}/test_predictions.csv", index_col=0, parse_dates=True)
        preds_test = {c: P[c].dropna() for c in P.columns}

def save_state():
    json.dump({"results": results,
               "feature_set": FEATURES,
               "split": {"train_end": TRAIN_END, "val_end": VAL_END}},
              open(CKPT, "w"), indent=2)
    if preds_test:
        pd.DataFrame({k: v.reindex(te.index) for k, v in preds_test.items()}).to_csv(
            f"{RES}/test_predictions.csv")

def report(name, r):
    m = r["test"]
    print(f"[{name}] test RMSE {m['rmse']:.4f}  MAE {m['mae']:.4f}  "
          f"MAPE {m['mape']:.2f}%  R2 {m['r2']:.4f}  (train {r['train_s']:.1f}s)")

print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")

# ----------------------------------------------------------------- baselines
if "seasonal_naive_24h" not in results:
    print("\n[1] Seasonal naive baselines")
    p24 = y.shift(24).loc[te.index]
    p168 = y.shift(168).loc[te.index]
    results["seasonal_naive_24h"] = {"test": metrics(y_te, p24), "train_s": 0.0}
    results["seasonal_naive_168h"] = {"test": metrics(y_te, p168), "train_s": 0.0}
    preds_test["seasonal_naive_24h"] = p24
    preds_test["seasonal_naive_168h"] = p168
    report("naive-24h", results["seasonal_naive_24h"])
    report("naive-168h", results["seasonal_naive_168h"])
    save_state()

# ----------------------------------------------------------------- ridge
if "ridge" not in results:
    print("\n[2] Ridge (cyclical encoding)")
    def cyc(df):
        Z = pd.DataFrame(index=df.index)
        a = lambda p, q: 2 * np.pi * df[p] / q
        Z["sin_h"], Z["cos_h"] = np.sin(a("hour", 24)), np.cos(a("hour", 24))
        Z["sin_d"], Z["cos_d"] = np.sin(a("dayofweek", 7)), np.cos(a("dayofweek", 7))
        Z["sin_m"], Z["cos_m"] = np.sin(a("month", 12)), np.cos(a("month", 12))
        Z["sin_dy"], Z["cos_dy"] = np.sin(a("dayofyear", 365.25)), np.cos(a("dayofyear", 365.25))
        for c in ["is_weekend", "lag_24", "lag_48", "lag_168",
                  "roll_mean_24", "roll_mean_168", "roll_std_24"]:
            Z[c] = df[c]
        return Z
    Z_tr, Z_va, Z_te = cyc(tr), cyc(va), cyc(te)
    best = (None, np.inf)
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=alpha).fit(Z_tr, y_tr)
        r = np.sqrt(mean_squared_error(y_va, m.predict(Z_va)))
        if r < best[1]:
            best = (alpha, r)
    t0 = time.time()
    ridge = Ridge(alpha=best[0]).fit(Z_tr, y_tr)
    train_s = time.time() - t0
    t0 = time.time()
    p = pd.Series(ridge.predict(Z_te), index=te.index)
    infer_ms = (time.time() - t0) * 1000 / len(te)
    results["ridge"] = {"test": metrics(y_te, p), "alpha": best[0],
                        "train_s": train_s, "infer_us_per_point": infer_ms * 1000}
    preds_test["ridge"] = p
    joblib.dump((ridge, list(Z_tr.columns)), f"{MOD}/ridge.joblib")
    report("ridge", results["ridge"])
    save_state()

# ----------------------------------------------------------------- holt-winters
def hw_forecast(series_all, target_index, seasonal="add", window=8 * 168):
    """Day-ahead forecasts: daily refit on trailing 8-week window.
    Fits on plain values (gappy index would break statsmodels' date handling)
    and stamps the 24 forecast steps with an explicit hourly range."""
    out = {}
    for d in sorted(set(target_index.normalize())):
        hist = series_all[series_all.index < d].iloc[-window:]
        try:
            m = ExponentialSmoothing(hist.values, trend="add", damped_trend=True,
                                     seasonal=seasonal, seasonal_periods=24,
                                     initialization_method="estimated").fit(optimized=True)
            fc = np.asarray(m.forecast(24))
            stamps = pd.date_range(d, periods=24, freq="h")
            for ts, v in zip(stamps, fc):
                if ts in target_index:
                    out[ts] = float(v)
        except Exception:
            pass
    return pd.Series(out)

if "holt_winters" not in results:
    print("\n[3] Holt-Winters ETS (rolling daily refits, 8-week window)")
    t0 = time.time()
    best_hw, best_r = None, np.inf
    for seas in ["add", "mul"]:
        pv = hw_forecast(y.loc[:VAL_END], va.index, seasonal=seas).reindex(va.index)
        common = pv.dropna().index
        r = np.sqrt(mean_squared_error(y_va[common], pv[common]))
        print(f"  seasonal={seas}: val RMSE {r:.4f} (n={len(common)})")
        if r < best_r:
            best_hw, best_r = seas, r
    p = hw_forecast(y, te.index, seasonal=best_hw).reindex(te.index).ffill()
    train_s = time.time() - t0
    t0 = time.time()
    infer_ms = (time.time() - t0) * 1000 / len(te)
    results["holt_winters"] = {"test": metrics(y_te, p), "seasonal": best_hw,
                               "train_s": train_s, "infer_us_per_point": infer_ms * 1000}
    preds_test["holt_winters"] = p
    report("holt_winters", results["holt_winters"])
    save_state()

# ----------------------------------------------------------------- random forest
if "random_forest" not in results:
    print("\n[4] Random Forest")
    Ftr, Fva, Fte = tr[FEATURES], va[FEATURES], te[FEATURES]
    best = (None, np.inf)
    for depth in [10, 18, None]:
        for leaf in [1, 4]:
            m = RandomForestRegressor(n_estimators=300, max_depth=depth,
                                      min_samples_leaf=leaf, n_jobs=2,
                                      random_state=SEED).fit(Ftr, y_tr)
            r = np.sqrt(mean_squared_error(y_va, m.predict(Fva)))
            if r < best[1]:
                best = ((depth, leaf), r)
            print(f"  depth={depth} leaf={leaf}: val RMSE {r:.4f}")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=best[0][0],
                               min_samples_leaf=best[0][1], n_jobs=2,
                               random_state=SEED).fit(Ftr, y_tr)
    train_s = time.time() - t0
    t0 = time.time()
    p = pd.Series(rf.predict(Fte), index=te.index)
    infer_ms = (time.time() - t0) * 1000 / len(te)
    results["random_forest"] = {"test": metrics(y_te, p),
                                "params": {"max_depth": best[0][0], "min_samples_leaf": best[0][1]},
                                "train_s": train_s, "infer_us_per_point": infer_ms * 1000}
    preds_test["random_forest"] = p
    joblib.dump((rf, FEATURES), f"{MOD}/random_forest.joblib")
    report("random_forest", results["random_forest"])
    save_state()

# ----------------------------------------------------------------- xgboost
if "xgboost" not in results:
    print("\n[5] XGBoost")
    Ftr, Fva, Fte = tr[FEATURES], va[FEATURES], te[FEATURES]
    best = (None, np.inf)
    for ne, md, lr in [(400, 4, 0.06), (400, 6, 0.06), (800, 4, 0.03), (800, 6, 0.03), (600, 5, 0.05)]:
        m = xgb.XGBRegressor(n_estimators=ne, max_depth=md, learning_rate=lr,
                             subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                             nthread=2, random_state=SEED, tree_method="hist")
        m.fit(Ftr, y_tr, eval_set=[(Fva, y_va)], verbose=False)
        r = np.sqrt(mean_squared_error(y_va, m.predict(Fva)))
        if r < best[1]:
            best = ((ne, md, lr), r)
        print(f"  n={ne} d={md} lr={lr}: val RMSE {r:.4f}")
    t0 = time.time()
    xgbm = xgb.XGBRegressor(n_estimators=best[0][0], max_depth=best[0][1],
                            learning_rate=best[0][2], subsample=0.9,
                            colsample_bytree=0.9, reg_lambda=1.0, nthread=2,
                            random_state=SEED, tree_method="hist").fit(Ftr, y_tr)
    train_s = time.time() - t0
    t0 = time.time()
    p = pd.Series(xgbm.predict(Fte), index=te.index)
    infer_ms = (time.time() - t0) * 1000 / len(te)
    results["xgboost"] = {"test": metrics(y_te, p),
                          "params": {"n_estimators": best[0][0], "max_depth": best[0][1], "lr": best[0][2]},
                          "train_s": train_s, "infer_us_per_point": infer_ms * 1000}
    preds_test["xgboost"] = p
    xgbm.save_model(f"{MOD}/xgboost.json")
    report("xgboost", results["xgboost"])
    save_state()

# ----------------------------------------------------------------- lightgbm
if "lightgbm" not in results:
    print("\n[6] LightGBM")
    Ftr, Fva, Fte = tr[FEATURES], va[FEATURES], te[FEATURES]
    best = (None, np.inf)
    for ne, nl, lr in [(400, 31, 0.06), (400, 63, 0.06), (800, 31, 0.03), (800, 63, 0.03), (600, 47, 0.05)]:
        m = lgb.LGBMRegressor(n_estimators=ne, num_leaves=nl, learning_rate=lr,
                              subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                              n_jobs=2, random_state=SEED, verbose=-1)
        m.fit(Ftr, y_tr, eval_set=[(Fva, y_va)])
        r = np.sqrt(mean_squared_error(y_va, m.predict(Fva)))
        if r < best[1]:
            best = ((ne, nl, lr), r)
        print(f"  n={ne} leaves={nl} lr={lr}: val RMSE {r:.4f}")
    t0 = time.time()
    lgbm = lgb.LGBMRegressor(n_estimators=best[0][0], num_leaves=best[0][1],
                             learning_rate=best[0][2], subsample=0.9,
                             colsample_bytree=0.9, reg_lambda=1.0, n_jobs=2,
                             random_state=SEED, verbose=-1).fit(Ftr, y_tr)
    train_s = time.time() - t0
    t0 = time.time()
    p = pd.Series(lgbm.predict(Fte), index=te.index)
    infer_ms = (time.time() - t0) * 1000 / len(te)
    results["lightgbm"] = {"test": metrics(y_te, p),
                           "params": {"n_estimators": best[0][0], "num_leaves": best[0][1], "lr": best[0][2]},
                           "train_s": train_s, "infer_us_per_point": infer_ms * 1000}
    preds_test["lightgbm"] = p
    joblib.dump((lgbm, FEATURES), f"{MOD}/lightgbm.joblib")
    report("lightgbm", results["lightgbm"])
    save_state()

# ----------------------------------------------------------------- LSTM
if "lstm" not in results:
    print("\n[7] LSTM (direct multi-step 168h -> 24h)")
    PAST, HORIZON = 168, 24
    mu, sd = float(y_tr.mean()), float(y_tr.std())
    y_all = (y - mu) / sd
    sin_h = np.sin(2 * np.pi * y.index.hour / 24).values
    cos_h = np.cos(2 * np.pi * y.index.hour / 24).values
    sin_d = np.sin(2 * np.pi * y.index.dayofweek / 7).values
    cos_d = np.cos(2 * np.pi * y.index.dayofweek / 7).values
    M = np.stack([y_all.values, sin_h, cos_h, sin_d, cos_d], axis=1)
    Yv = y_all.values
    pos = {ts: i for i, ts in enumerate(y.index)}

    def make_windows(idx_range):
        Xs, Ys, TSs = [], [], []
        for ts in idx_range:
            i = pos[ts]
            j = i - HORIZON          # window end 24h before target
            if j - PAST + 1 >= 0 and i < len(y):
                Xs.append(M[j - PAST + 1: j + 1])
                Ys.append(Yv[j + 1: j + HORIZON + 1])
                TSs.append(ts)
        return (torch.tensor(np.array(Xs), dtype=torch.float32),
                torch.tensor(np.array(Ys), dtype=torch.float32), TSs)

    Xtr, Ytr, _ = make_windows(tr.index)
    Xva, Yva, _ = make_windows(va.index)
    Xte, _, TSte = make_windows(te.index)
    print(f"  tensors: train {tuple(Xtr.shape)} val {tuple(Xva.shape)} test {tuple(Xte.shape)}")

    class LSTMForecaster(nn.Module):
        def __init__(self, hidden=64, layers=2, in_dim=5, horizon=24):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=0.15)
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, horizon))
        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1, :])

    model = LSTMForecaster()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.SmoothL1Loss()
    trl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(Xva, Yva), batch_size=256)
    # epoch-level resume state
    CK = f"{MOD}/lstm_train_state.pt"
    start_ep, best_vl, wait, hist, best_state = 0, np.inf, 0, {"train": [], "val": []}, None
    if os.path.exists(CK):
        st = torch.load(CK, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_ep = st["epoch"] + 1
        best_vl, wait, hist = st["best_vl"], st["wait"], st["hist"]
        best_state = st["best_state"]
        print(f"  resuming from epoch {start_ep + 1} (best val {best_vl:.4f})")
    t0 = time.time()
    for ep in range(start_ep, 40):
        model.train(); tl = []
        for xb, yb in trl:
            opt.zero_grad(); l = lossf(model(xb), yb)
            l.backward(); opt.step(); tl.append(l.item())
        model.eval(); vl = []
        with torch.no_grad():
            for xb, yb in val:
                vl.append(lossf(model(xb), yb).item())
        tl, vl = float(np.mean(tl)), float(np.mean(vl))
        hist["train"].append(tl); hist["val"].append(vl)
        if vl < best_vl:
            best_vl, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        print(f"  epoch {ep+1:02d}  train {tl:.4f}  val {vl:.4f}{' *' if wait == 0 else ''}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "best_vl": best_vl, "wait": wait, "hist": hist,
                    "best_state": best_state}, CK)
        if wait >= 6:
            print("  early stop")
            break
    train_s = time.time() - t0
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd,
                "config": {"hidden": 64, "layers": 2, "in_dim": 5, "horizon": 24, "past": 168}},
               f"{MOD}/lstm_forecaster.pt")
    os.remove(CK)
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        phat = model(Xte).numpy() * sd + mu
    infer_ms = (time.time() - t0) * 1000 / len(Xte)
    p = pd.Series(phat[:, -1], index=TSte).sort_index()   # strict 24h-ahead step
    results["lstm"] = {"test": metrics(y_te.reindex(p.index), p),
                       "params": {"hidden": 64, "layers": 2, "epochs_run": len(hist["train"]),
                                  "past": 168, "horizon": 24},
                       "train_s": train_s, "infer_us_per_point": infer_ms * 1000}
    preds_test["lstm"] = p
    json.dump(hist, open(f"{RES}/lstm_training_history.json", "w"))
    report("lstm", results["lstm"])
    save_state()

# ----------------------------------------------------------------- summary
print("\n" + "=" * 70)
print("TEST RESULTS (strict 24h-ahead day-ahead task)")
print("=" * 70)
rows = []
for name, r in results.items():
    m = r["test"]
    skill = 1 - m["rmse"] / results["seasonal_naive_24h"]["test"]["rmse"]
    rows.append({"model": name, "rmse": m["rmse"], "mae": m["mae"],
                 "mape": m["mape"], "r2": m["r2"], "skill_vs_naive": skill})
df = pd.DataFrame(rows).sort_values("rmse")
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
d = json.load(open(CKPT))
d["summary_table"] = df.to_dict(orient="records")
json.dump(d, open(CKPT, "w"), indent=2)
print("\nDone. Best model:", df.iloc[0]["model"])

"""
Anomaly detection: evaluation on synthetic injected anomalies + real case study.  (v2)

Injection: ~2% of TEST *hours* corrupted (spikes / level shifts / dropouts),
lag features recomputed from the contaminated series (realistic protocol).

Detectors: residual z-score (XGBoost), Isolation Forest, One-Class SVM, LSTM-AE.
Scoring:   LSTM-AE error is localized per hour by averaging the squared error of
           every window that contains that hour (not just the window end).
Metrics:   ROC-AUC, PR-AUC, P/R/F1 at the actual prevalence operating point,
           per-type recall (spike / shift / dropout). Case study on clean test.
"""
import json, os, time, warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)
import xgboost as xgb

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
CKPT = f"{RES}/anomaly_metrics.json"

h = pd.read_csv(f"{DATA}/hourly_full.csv", index_col=0, parse_dates=True)
y = h["global_active_power"]
TRAIN_END, VAL_END = "2009-06-30 23:00", "2010-03-31 23:00"
test_idx_all = y.index[y.index > VAL_END]

F_COLS = ["hour", "dayofweek", "month", "is_weekend", "dayofyear",
          "lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_mean_168", "roll_std_24"]
IF_FEATS = ["gap", "lag_24", "lag_168", "roll_mean_24", "roll_std_24", "hour", "dayofweek"]

def build_features(series):
    X = pd.DataFrame(index=series.index)
    X["gap"] = series
    X["hour"] = series.index.hour
    X["dayofweek"] = series.index.dayofweek
    X["month"] = series.index.month
    X["is_weekend"] = (series.index.dayofweek >= 5).astype(int)
    X["lag_24"] = series.shift(24)
    X["lag_48"] = series.shift(48)
    X["lag_168"] = series.shift(168)
    X["roll_mean_24"] = series.shift(24).rolling(24).mean()
    X["roll_mean_168"] = series.shift(24).rolling(168).mean()
    X["roll_std_24"] = series.shift(24).rolling(24).std()
    X["dayofyear"] = series.index.dayofyear
    return X.dropna()

X_full = build_features(y)
tr = X_full[X_full.index <= TRAIN_END]
va = X_full[(X_full.index > TRAIN_END) & (X_full.index <= VAL_END)]

# ================================================================ injection
def inject(series, seed=42, rate=0.02):
    """Corrupt ~rate of test HOURS. Spikes 40%, shifts 30%, dropouts 30% of budget."""
    r = np.random.default_rng(seed)
    s = series.copy()
    labels = pd.Series(0, index=series.index, dtype=int)
    atype = pd.Series("", index=series.index, dtype=object)
    test_pos = np.where(series.index > VAL_END)[0]
    budget = int(len(test_pos) * rate)
    used = set()
    first_test = min(test_pos)
    # --- spikes: single hours (40% of budget)
    n_spike = int(budget * 0.4)
    for p in r.choice(test_pos, size=n_spike, replace=False):
        s.iloc[p] = min(s.iloc[p] * r.uniform(2.5, 4.0) + 1.5, 8.0)
        labels.iloc[p] = 1; atype.iloc[p] = "spike"; used.add(int(p))
    # --- level shifts: 2-6h blocks (30% of budget)
    shift_budget = int(budget * 0.3)
    placed = 0
    while placed < shift_budget:
        p = int(r.choice(test_pos)); L = int(r.integers(2, 7))
        block = list(range(p, min(p + L, len(s))))
        if not block or any(b in used for b in block) or p < first_test:
            continue
        f = r.uniform(1.6, 2.5)
        for b in block:
            s.iloc[b] = min(s.iloc[b] * f, 8.0)
            labels.iloc[b] = 1; atype.iloc[b] = "shift"
        used.update(block); placed += len(block)
    # --- dropouts: 4-12h near-zero blocks (remaining budget)
    drop_budget = budget - n_spike - placed
    placed = 0
    while placed < drop_budget:
        p = int(r.choice(test_pos)); L = int(r.integers(4, 13))
        block = list(range(p, min(p + L, len(s))))
        if not block or any(b in used for b in block) or p < first_test:
            continue
        for b in block:
            s.iloc[b] = r.uniform(0.02, 0.10)
            labels.iloc[b] = 1; atype.iloc[b] = "dropout"
        used.update(block); placed += len(block)
    return s, labels, atype

y_corrupt, y_lab, y_type = inject(y, seed=SEED)
X_corrupt = build_features(y_corrupt)
te_corrupt = X_corrupt[X_corrupt.index > VAL_END]
n_anom = int(y_lab.sum())
rate_actual = n_anom / len(test_idx_all)
print(f"injected: {n_anom} hours ({rate_actual*100:.2f}% of test) | "
      f"spike={int((y_type=='spike').sum())} shift={int((y_type=='shift').sum())} "
      f"dropout={int((y_type=='dropout').sum())}")

eval_idx = te_corrupt.index
lab = y_lab.reindex(eval_idx).values.astype(int)
typ = y_type.reindex(eval_idx).values
Q = 1 - rate_actual  # operating point = actual prevalence

def prf(scores, labels, types, q):
    pred = (scores >= np.quantile(scores, q)).astype(int)
    res = {"precision": float(precision_score(labels, pred, zero_division=0)),
           "recall": float(recall_score(labels, pred, zero_division=0)),
           "f1": float(f1_score(labels, pred, zero_division=0)),
           "roc_auc": float(roc_auc_score(labels, scores)),
           "pr_auc": float(average_precision_score(labels, scores))}
    for t in ["spike", "shift", "dropout"]:
        m = types == t
        res[f"recall_{t}"] = float((pred[m] == 1).mean()) if m.sum() else None
    return res

out = {}
if os.path.exists(CKPT):
    out = json.load(open(CKPT)).get("results", {})
    if out.get("_protocol") != "v2":
        out = {}  # protocol changed; recompute
    print(f"RESUME anomaly: {list(out)}")

# ================================================================ A) residual z-score
if "residual_zscore" not in out:
    print("\n[A] Residual z-score (XGBoost forecaster)")
    model = xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                             subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                             nthread=2, random_state=SEED, tree_method="hist")
    model.fit(tr[F_COLS], tr["gap"], eval_set=[(va[F_COLS], va["gap"])], verbose=False)
    sigma = float(np.std(va["gap"].values - model.predict(va[F_COLS])))
    te_pred = model.predict(te_corrupt[F_COLS])
    z = np.abs(te_corrupt["gap"].values - te_pred) / sigma
    out["residual_zscore"] = prf(z, lab, typ, Q)
    pred3 = (z > 3).astype(int)
    out["residual_zscore"]["f1_3sigma"] = float(f1_score(lab, pred3, zero_division=0))
    out["residual_zscore"]["precision_3sigma"] = float(precision_score(lab, pred3, zero_division=0))
    out["residual_zscore"]["recall_3sigma"] = float(recall_score(lab, pred3, zero_division=0))
    print("  ", {k: round(v, 4) for k, v in out["residual_zscore"].items() if v is not None})
    pd.DataFrame({"timestamp": eval_idx, "z": z, "label": lab}).to_csv(
        f"{RES}/anomaly_residual_scores.csv", index=False)

# ================================================================ B) Isolation Forest
if "isolation_forest" not in out:
    print("\n[B] Isolation Forest")
    sc = StandardScaler().fit(tr[IF_FEATS])
    t0 = time.time()
    iso = IsolationForest(n_estimators=300, contamination=float(rate_actual),
                          random_state=SEED, n_jobs=2).fit(sc.transform(tr[IF_FEATS]))
    train_s = time.time() - t0
    raw = -iso.score_samples(sc.transform(te_corrupt[IF_FEATS]))
    out["isolation_forest"] = prf(raw, lab, typ, Q)
    out["isolation_forest"]["train_s"] = train_s
    print("  ", {k: round(v, 4) for k, v in out["isolation_forest"].items() if v is not None})

# ================================================================ C) One-Class SVM
if "one_class_svm" not in out:
    print("\n[C] One-Class SVM (RBF)")
    sc2 = StandardScaler().fit(tr[IF_FEATS])
    trs = tr[IF_FEATS].sample(n=min(12000, len(tr)), random_state=SEED)
    t0 = time.time()
    oc = OneClassSVM(kernel="rbf", nu=float(rate_actual), gamma="scale").fit(sc2.transform(trs))
    train_s = time.time() - t0
    raw = -oc.decision_function(sc2.transform(te_corrupt[IF_FEATS])).ravel()
    out["one_class_svm"] = prf(raw, lab, typ, Q)
    out["one_class_svm"]["train_s"] = train_s
    print("  ", {k: round(v, 4) for k, v in out["one_class_svm"].items() if v is not None})

# ================================================================ D) LSTM autoencoder
if "lstm_autoencoder" not in out:
    print("\n[D] LSTM autoencoder (24h windows, localized scoring)")
    W = 24
    mu, sd = float(tr["gap"].mean()), float(tr["gap"].std())
    zv_clean = ((y - mu) / sd)
    zv_corrupt = ((y_corrupt - mu) / sd)
    pos_of = {ts: i for i, ts in enumerate(y.index)}
    arr = zv_corrupt.values

    def windows(idxs):
        Xs, TSs, P0 = [], [], []
        for ts in idxs:
            p = pos_of[ts]
            if p - W + 1 >= 0:
                Xs.append(arr[p - W + 1: p + 1]); TSs.append(ts); P0.append(p - W + 1)
        return torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1), TSs, P0

    Xtr, _, _ = windows(tr.index)
    Xva, _, _ = windows(va.index)
    Xte, TS_te, P0_te = windows(te_corrupt.index)
    print(f"  AE windows: train {len(Xtr)} val {len(Xva)} test {len(Xte)}")

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

    ae = LSTMAE()
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    CKAE = f"{MOD}/lstm_ae_state.pt"
    start_ep, best_vl, wait = 0, np.inf, 0
    if os.path.exists(CKAE):
        st = torch.load(CKAE, weights_only=False)
        ae.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_ep, best_vl, wait = st["epoch"] + 1, st["wait"], st["best_vl"]
        print(f"  resume from epoch {start_ep+1}")
    trl = DataLoader(TensorDataset(Xtr, Xtr), batch_size=128, shuffle=True)
    val = DataLoader(TensorDataset(Xva, Xva), batch_size=512)
    t0 = time.time()
    for ep in range(start_ep, 30):
        ae.train(); tl = []
        for xb, _ in trl:
            opt.zero_grad(); l = lossf(ae(xb), xb)
            l.backward(); opt.step(); tl.append(l.item())
        ae.eval(); vl = []
        with torch.no_grad():
            for xb, _ in val:
                vl.append(lossf(ae(xb), xb).item())
        tl, vl = float(np.mean(tl)), float(np.mean(vl))
        if vl < best_vl:
            best_vl, wait = vl, 0
        else:
            wait += 1
        print(f"  epoch {ep+1:02d}  train {tl:.5f}  val {vl:.5f}")
        torch.save({"model": ae.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "best_vl": best_vl, "wait": wait}, CKAE)
        if wait >= 5:
            break
    train_s = time.time() - t0
    ae.eval()
    with torch.no_grad():
        rec = ae(Xte).squeeze(-1).numpy()
    err_mat = (rec - Xte.squeeze(-1).numpy()) ** 2          # (n_windows, W)
    # localize: hour at absolute pos p appears in windows with P0 in [p-W+1, p]
    n = len(arr)
    hour_err = np.zeros(n); hour_cnt = np.zeros(n)
    for wi, p0 in enumerate(P0_te):
        for k in range(W):
            hour_err[p0 + k] += err_mat[wi, k]
            hour_cnt[p0 + k] += 1
    hour_score = np.where(hour_cnt > 0, hour_err / np.maximum(hour_cnt, 1), 0.0)
    score_series = pd.Series(hour_score, index=y.index).reindex(eval_idx).fillna(0)
    out["lstm_autoencoder"] = prf(score_series.values, lab, typ, Q)
    out["lstm_autoencoder"]["train_s"] = train_s
    torch.save({"state_dict": ae.state_dict(), "mu": mu, "sd": sd,
                "config": {"hidden": 32, "w": W}}, f"{MOD}/lstm_autoencoder.pt")
    os.remove(CKAE)
    print("  ", {k: round(v, 4) for k, v in out["lstm_autoencoder"].items() if v is not None})

# ================================================================ save + summary
out["_protocol"] = "v2"
print("\n" + "=" * 70)
print(f"ANOMALY DETECTION (injection {rate_actual*100:.2f}%, operating point = prevalence)")
print("=" * 70)
rows = [{"detector": k, **{m: (round(v, 4) if isinstance(v, float) else v)
                           for m, v in r.items() if m != "train_s"}}
        for k, r in out.items() if k != "_protocol"]
df = pd.DataFrame(rows).sort_values("f1", ascending=False)
print(df.to_string(index=False))
json.dump({"results": out, "n_anomalies": n_anom, "n_test": len(test_idx_all),
           "injection_rate": float(rate_actual),
           "n_spike": int((y_type == "spike").sum()),
           "n_shift": int((y_type == "shift").sum()),
           "n_dropout": int((y_type == "dropout").sum())},
          open(CKPT, "w"), indent=2)
print("saved", CKPT)

# ================================================================ case study (clean test)
print("\nCase study: residual z-score events on CLEAN test set")
mdl = xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                       subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                       nthread=2, random_state=SEED, tree_method="hist")
mdl.fit(tr[F_COLS], tr["gap"], verbose=False)
sigma = float(np.std(va["gap"].values - mdl.predict(va[F_COLS])))
teF = X_full[X_full.index > VAL_END]
te_pred = pd.Series(mdl.predict(teF[F_COLS]), index=teF.index)
zz = ((teF["gap"] - te_pred).abs() / sigma)
flag = zz > 3
print(f"flagged hours (|z|>3): {int(flag.sum())} ({flag.sum()/len(flag)*100:.2f}%)")
events = []
idxs = list(zz.index[flag])
if idxs:
    start, prev = idxs[0], idxs[0]
    for ts in idxs[1:] + [None]:
        if ts is not None and (ts - prev).total_seconds() <= 3600 * 2:
            prev = ts
        else:
            seg = zz.loc[start:prev]
            j = seg.idxmax()
            events.append({"start": str(start), "end": str(prev),
                           "n_hours": int((prev - start).total_seconds() // 3600 + 1),
                           "max_z": float(seg.max()),
                           "actual_kw": float(teF.loc[j, "gap"]),
                           "predicted_kw": float(te_pred[j]),
                           "direction": "high" if teF.loc[j, "gap"] > te_pred[j] else "low"})
            if ts is not None:
                start, prev = ts, ts
events = sorted(events, key=lambda e: -e["max_z"])[:12]
json.dump(events, open(f"{RES}/anomaly_case_study.json", "w"), indent=2)
for e in events[:10]:
    print(f"  {e['start']} .. {e['end']}  {e['n_hours']}h  z={e['max_z']:.1f}  "
          f"actual {e['actual_kw']:.2f} vs pred {e['predicted_kw']:.2f} kW [{e['direction']}]")
print("saved results/anomaly_case_study.json")

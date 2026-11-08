"""Publication figures (Okabe-Ito colorblind-safe palette, vector PDF output)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

FIG = "/home/z/my-project/figures"
RES = "/home/z/my-project/results"
DATA = "/home/z/my-project/data"
import os
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito palette
C_BLUE, C_ORANGE, C_GREEN, C_RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
C_PURPLE, C_CYAN, C_GRAY, C_BLACK = "#CC79A7", "#56B4E9", "#999999", "#000000"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "legend.frameon": False,
})

h = pd.read_csv(f"{DATA}/hourly_full.csv", index_col=0, parse_dates=True)
y = h["global_active_power"]
fm = json.load(open(f"{RES}/forecast_metrics.json"))
preds = pd.read_csv(f"{RES}/test_predictions.csv", index_col=0, parse_dates=True)
am = json.load(open(f"{RES}/anomaly_metrics.json"))
hist = json.load(open(f"{RES}/lstm_training_history.json"))
TRAIN_END, VAL_END = pd.Timestamp("2009-06-30 23:00"), pd.Timestamp("2010-03-31 23:00")

# ================================================================ Fig 1: dataset overview
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.1), constrained_layout=True)
ax = axes[0]
daily = y.resample("D").mean() * 24  # kWh per day (mean kW x 24 h)
ax.fill_between(daily.index, daily.values, color=C_BLUE, alpha=0.25, linewidth=0)
ax.plot(daily.index, daily.values, color=C_BLUE, lw=0.7)
for t, lab, col in [(TRAIN_END, "Train", C_GREEN), (VAL_END, "Validation", C_ORANGE)]:
    ax.axvline(t, color=col, ls="--", lw=1.2)
ax.axvspan(VAL_END, daily.index[-1], color=C_RED, alpha=0.06)
yt = daily.quantile(0.98)
ax.text(pd.Timestamp("2008-01-15"), yt * 0.92, "Train", color=C_GREEN, fontsize=9)
ax.text(pd.Timestamp("2009-08-15"), yt * 0.92, "Val", color=C_ORANGE, fontsize=9)
ax.text(pd.Timestamp("2010-05-20"), yt * 0.92, "Test", color=C_RED, fontsize=9)
ax.set_ylabel("Daily energy consumption (kWh)")
ax.set_xlabel("Date")
ax.set_title("(a) Daily household energy consumption, 2006–2010")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax = axes[1]
prof_wd = y[y.index.dayofweek < 5].groupby(y[y.index.dayofweek < 5].index.hour).mean()
prof_we = y[y.index.dayofweek >= 5].groupby(y[y.index.dayofweek >= 5].index.hour).mean()
ax.plot(prof_wd.index, prof_wd.values, color=C_BLUE, lw=2, label="Weekdays")
ax.plot(prof_we.index, prof_we.values, color=C_ORANGE, lw=2, label="Weekend")
ax.fill_between(prof_wd.index, prof_wd.values, alpha=0.15, color=C_BLUE)
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean active power (kW)")
ax.set_title("(b) Average daily profile")
ax.legend()
fig.savefig(f"{FIG}/fig1_dataset_overview.pdf")
plt.close(fig)
print("fig1 done")

# ================================================================ Fig 2: seasonality
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), constrained_layout=True)
ax = axes[0]
piv = y.groupby([y.index.month, y.index.hour]).mean().unstack()
im = ax.imshow(piv.values, aspect="auto", cmap="YlGnBu",
               extent=[0, 24, 12.5, 0.5], origin="upper")
ax.set_xlabel("Hour of day"); ax.set_ylabel("Month")
ax.set_yticks(range(1, 13))
ax.set_xticks(range(0, 24, 3))
ax.set_title("(a) Mean power by month and hour (kW)")
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.ax.set_ylabel("kW", rotation=270, labelpad=8)

ax = axes[1]
from statsmodels.tsa.seasonal import STL
seg = y.loc["2010-01-01":"2010-02-28"].asfreq("h").interpolate()
stl = STL(seg, period=24, robust=True).fit()
ax.plot(seg.index, seg.values, color=C_GRAY, lw=0.6, label="Observed")
ax.plot(stl.trend.index, stl.trend.values, color=C_RED, lw=1.8, label="Trend")
ax.plot(stl.seasonal.index, stl.seasonal.values + seg.mean(), color=C_BLUE, lw=0.9,
        alpha=0.85, label="Daily seasonal (offset)")
ax.set_ylabel("Power (kW)")
ax.set_xlabel("Date")
ax.set_title("(b) STL decomposition (Jan–Feb 2010)")
ax.legend(loc="upper left", ncols=3, fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.savefig(f"{FIG}/fig2_seasonality.pdf")
plt.close(fig)
print("fig2 done")

# ================================================================ Fig 3: model comparison
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), constrained_layout=True)
tab = pd.DataFrame(fm["summary_table"]).sort_values("rmse")
nice = {"seasonal_naive_24h": "Naive-24h", "seasonal_naive_168h": "Naive-168h",
        "ridge": "Ridge", "holt_winters": "Holt-Winters", "random_forest": "Random Forest",
        "xgboost": "XGBoost", "lightgbm": "LightGBM", "lstm": "LSTM"}
colors = [C_RED if n == "xgboost" else (C_BLUE if sk > 0 else C_GRAY)
          for n, sk in zip(tab["model"], tab["skill_vs_naive"])]
ax = axes[0]
b = ax.barh([nice[m] for m in tab["model"]], tab["rmse"], color=colors, height=0.62)
for r, v in zip(b, tab["rmse"]):
    ax.text(v + 0.006, r.get_y() + r.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("Test RMSE (kW) — lower is better")
ax.set_title("(a) Forecast accuracy (24 h ahead)")
ax.set_xlim(0, max(tab["rmse"]) * 1.14)

ax = axes[1]
names_b = [nice[m].replace("Random Forest", "Rand. Forest").replace("Holt-Winters", "Holt-Wint.") for m in tab["model"]]
ax.barh(names_b, tab["skill_vs_naive"] * 100, color=colors, height=0.62)
ax.axvline(0, color=C_BLACK, lw=0.8)
ax.set_xlabel("Forecast skill vs naive-24h (%)")
ax.set_title("(b) Improvement over seasonal naive")
ax.tick_params(axis="y", labelsize=8)
for i, v in enumerate(tab["skill_vs_naive"] * 100):
    ax.text(v + (0.6 if v >= 0 else -0.6), i, f"{v:+.1f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8)
ax.set_xlim(min(tab["skill_vs_naive"]) * 100 - 9, max(tab["skill_vs_naive"]) * 100 + 9)
fig.savefig(f"{FIG}/fig3_model_comparison.pdf")
plt.close(fig)
print("fig3 done")

# ================================================================ Fig 4: forecast quality week
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), constrained_layout=True)
wk_start, wk_end = "2010-07-05", "2010-07-12"
mask = (preds.index >= wk_start) & (preds.index < wk_end)
actual_aligned = h["global_active_power"].reindex(preds.index)
ax = axes[0]
ax.plot(preds.index[mask], preds.loc[mask, "xgboost"], color=C_RED, lw=1.6,
        label="XGBoost forecast")
ax.plot(preds.index[mask], preds.loc[mask, "lstm"], color=C_PURPLE, lw=1.3, ls="-.",
        label="LSTM forecast")
ax.plot(preds.index[mask], preds.loc[mask, "seasonal_naive_24h"], color=C_GRAY, lw=1.0,
        ls=":", label="Naive-24h")
ax.plot(preds.index[mask], actual_aligned[mask],
        color=C_BLACK, lw=1.1, alpha=0.75, label="Actual")
ax.set_ylabel("Active power (kW)")
ax.set_xlabel("Date (test period)")
ax.set_title("(a) Day-ahead forecasts vs actuals (one week)")
handles_a, labels_a = ax.get_legend_handles_labels()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

ax = axes[1]
te_true = h["global_active_power"].reindex(preds.index)
err_xgb = (preds["xgboost"] - te_true).dropna()
err_naive = (preds["seasonal_naive_24h"] - te_true).dropna()
hours = te_true.dropna().index.hour
d1 = pd.DataFrame({"e": err_xgb, "h": hours}).dropna()
d2 = pd.DataFrame({"e": err_naive, "h": hours}).dropna()
bx1 = [d1[d1.h == hh]["e"].values for hh in range(24)]
bx2 = [d2[d2.h == hh]["e"].values for hh in range(24)]
bp = ax.boxplot(bx2, positions=np.arange(24) - 0.2, widths=0.35, patch_artist=True,
                showfliers=False, medianprops=dict(color=C_BLACK, lw=0.9),
                boxprops=dict(facecolor="#E0E0E0", edgecolor=C_GRAY),
                whiskerprops=dict(color=C_GRAY), capprops=dict(color=C_GRAY))
bp = ax.boxplot(bx1, positions=np.arange(24) + 0.2, widths=0.35, patch_artist=True,
                showfliers=False, medianprops=dict(color=C_BLACK, lw=0.9),
                boxprops=dict(facecolor="#F6C6AD", edgecolor=C_RED),
                whiskerprops=dict(color=C_RED), capprops=dict(color=C_RED))
ax.axhline(0, color=C_BLACK, lw=0.8)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{hh:02d}" for hh in range(0, 24, 2)], fontsize=7.5)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Forecast error (kW)")
ax.set_title("(b) Error distribution by hour")
ax.legend(handles=[Patch(facecolor="#F6C6AD", edgecolor=C_RED, label="XGBoost"),
                   Patch(facecolor="#E0E0E0", edgecolor=C_GRAY, label="Naive-24h")],
          loc="upper left", fontsize=8)
fig.legend(handles=handles_a, labels=labels_a, loc="outside upper center",
           ncol=4, fontsize=8, columnspacing=1.4)
fig.savefig(f"{FIG}/fig4_forecast_quality.pdf")
plt.close(fig)
print("fig4 done")

# ================================================================ Fig 5: importance + LSTM curves
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), constrained_layout=True)
ax = axes[0]
import xgboost as xgb
mdl = xgb.XGBRegressor()
mdl.load_model("/home/z/my-project/models/xgboost.json")
imp = pd.Series(mdl.get_booster().get_score(importance_type="gain")).sort_values()
feat_nice = {"lag_24": "lag 24 h", "lag_168": "lag 168 h", "lag_48": "lag 48 h",
             "roll_mean_24": "roll mean 24 h", "roll_mean_168": "roll mean 168 h",
             "roll_std_24": "roll std 24 h", "hour": "hour of day", "dayofweek": "day of week",
             "month": "month", "is_weekend": "weekend flag", "dayofyear": "day of year"}
ax.barh([feat_nice.get(k, k) for k in imp.index], imp.values, color=C_BLUE, height=0.62)
ax.set_xlabel("XGBoost split gain importance")
ax.set_title("(a) Feature importance (XGBoost)")

ax = axes[1]
ep = np.arange(1, len(hist["train"]) + 1)
ax.plot(ep, hist["train"], color=C_BLUE, lw=1.8, label="Train loss")
ax.plot(ep, hist["val"], color=C_RED, lw=1.8, label="Validation loss")
best = int(np.argmin(hist["val"])) + 1
ax.axvline(best, color=C_GRAY, ls="--", lw=1)
ax.text(best + 0.3, max(hist["train"]) * 0.97, f"best epoch = {best}", fontsize=8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Smooth-L1 loss (standardised)")
ax.set_title("(b) LSTM training dynamics")
ax.legend()
fig.savefig(f"{FIG}/fig5_importance_lstm.pdf")
plt.close(fig)
print("fig5 done")

# ================================================================ Fig 6: anomaly ROC + per-type recall
from sklearn.metrics import roc_curve, precision_recall_curve
sc = pd.read_csv(f"{RES}/anomaly_residual_scores.csv", parse_dates=["timestamp"])
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), constrained_layout=True)
ax = axes[0]
# rebuild scores for all detectors from stored metrics file? ROC needs raw scores;
# recompute quickly for IF/OCSVM/AE would need models -> instead re-run scoring from saved artifacts.
# We saved only residual scores; recompute others here.
print("  recomputing detector scores for ROC curves ...")
TRAIN_END_S, VAL_END_S = "2009-06-30 23:00", "2010-03-31 23:00"
import joblib
from sklearn.ensemble import IsolationForest as IF_
from sklearn.svm import OneClassSVM as OCSVM_
from sklearn.preprocessing import StandardScaler

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

IF_FEATS = ["gap", "lag_24", "lag_168", "roll_mean_24", "roll_std_24", "hour", "dayofweek"]
F_COLS = ["hour", "dayofweek", "month", "is_weekend", "dayofyear",
          "lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_mean_168", "roll_std_24"]
X_full = build_features(y)
trF = X_full[X_full.index <= TRAIN_END_S]
vaF = X_full.query(f"'{TRAIN_END_S}' < index <= '{VAL_END_S}'")

# reproduce the SAME injected series (seed 42, same code path as 03)
r = np.random.default_rng(42)
s = y.copy()
labels = pd.Series(0, index=y.index, dtype=int)
atypes = pd.Series("", index=y.index, dtype=object)
test_pos = np.where(y.index > VAL_END_S)[0]
budget = int(len(test_pos) * 0.02); used = set()
n_spike = int(budget * 0.4)
for p in r.choice(test_pos, size=n_spike, replace=False):
    s.iloc[p] = min(s.iloc[p] * r.uniform(2.5, 4.0) + 1.5, 8.0)
    labels.iloc[p] = 1; atypes.iloc[p] = "spike"; used.add(int(p))
sb = int(budget * 0.3); placed = 0
while placed < sb:
    p = int(r.choice(test_pos)); L = int(r.integers(2, 7))
    block = list(range(p, min(p + L, len(s))))
    if not block or any(b in used for b in block) or p < min(test_pos):
        continue
    f = r.uniform(1.6, 2.5)
    for b in block:
        s.iloc[b] = min(s.iloc[b] * f, 8.0)
        labels.iloc[b] = 1; atypes.iloc[b] = "shift"
    used.update(block); placed += len(block)
db = budget - n_spike - placed; placed = 0
while placed < db:
    p = int(r.choice(test_pos)); L = int(r.integers(4, 13))
    block = list(range(p, min(p + L, len(s))))
    if not block or any(b in used for b in block) or p < min(test_pos):
        continue
    for b in block:
        s.iloc[b] = r.uniform(0.02, 0.10)
        labels.iloc[b] = 1; atypes.iloc[b] = "dropout"
    used.update(block); placed += len(block)
X_cor = build_features(s)
te_cor = X_cor[X_cor.index > VAL_END_S]
lab_te = labels.reindex(te_cor.index).values.astype(int)
typ_te = atypes.reindex(te_cor.index).values

# z-score scores (same as stored)
xg = xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03,
                      subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                      nthread=2, random_state=42, tree_method="hist")
xg.fit(trF[F_COLS], trF["gap"], verbose=False)
sigma = float(np.std(vaF["gap"].values - xg.predict(vaF[F_COLS])))
z_sc = np.abs(te_cor["gap"].values - xg.predict(te_cor[F_COLS])) / sigma

sc1 = StandardScaler().fit(trF[IF_FEATS])
iso = IF_(n_estimators=300, contamination=0.02, random_state=42, n_jobs=2).fit(sc1.transform(trF[IF_FEATS]))
iso_sc = -iso.score_samples(sc1.transform(te_cor[IF_FEATS]))
trs = trF[IF_FEATS].sample(n=min(12000, len(trF)), random_state=42)
sc2 = StandardScaler().fit(trF[IF_FEATS])
oc = OCSVM_(kernel="rbf", nu=0.02, gamma="scale").fit(sc2.transform(trs))
svm_sc = -oc.decision_function(sc2.transform(te_cor[IF_FEATS])).ravel()

import torch, torch.nn as nn
ck = torch.load("/home/z/my-project/models/lstm_autoencoder.pt", weights_only=False)
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
ae = LSTMAE(); ae.load_state_dict(ck["state_dict"]); ae.eval()
W = 24
mu, sd = ck["mu"], ck["sd"]
arr = ((s - mu) / sd).values
pos_of = {ts: i for i, ts in enumerate(y.index)}
Xs, P0 = [], []
for ts in te_cor.index:
    p = pos_of[ts]
    if p - W + 1 >= 0:
        Xs.append(arr[p - W + 1: p + 1]); P0.append(p - W + 1)
Xt = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1)
with torch.no_grad():
    rec = ae(Xt).squeeze(-1).numpy()
errm = (rec - Xt.squeeze(-1).numpy()) ** 2
n = len(arr); he = np.zeros(n); hc = np.zeros(n)
for wi, p0 in enumerate(P0):
    for k in range(W):
        he[p0 + k] += errm[wi, k]; hc[p0 + k] += 1
ae_sc_all = np.where(hc > 0, he / np.maximum(hc, 1), 0)
ae_sc = pd.Series(ae_sc_all, index=y.index).reindex(te_cor.index).fillna(0).values

for name, sco, col in [("Residual z-score (XGBoost)", z_sc, C_RED),
                       ("Isolation Forest", iso_sc, C_BLUE),
                       ("One-Class SVM", svm_sc, C_GREEN),
                       ("LSTM autoencoder", ae_sc, C_PURPLE)]:
    fpr, tpr, _ = roc_curve(lab_te, sco)
    aucv = am["results"][{"Residual z-score (XGBoost)": "residual_zscore",
                          "Isolation Forest": "isolation_forest",
                          "One-Class SVM": "one_class_svm",
                          "LSTM autoencoder": "lstm_autoencoder"}[name]]["roc_auc"]
    ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name} (AUC={aucv:.3f})")
ax.plot([0, 1], [0, 1], color=C_GRAY, ls=":", lw=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("(a) ROC curves — injected anomaly benchmark")
ax.legend(fontsize=7.5, loc="lower right")

ax = axes[1]
dets = ["residual_zscore", "isolation_forest", "one_class_svm", "lstm_autoencoder"]
names = ["Residual\nz-score", "Isolation\nForest", "One-Class\nSVM", "LSTM\nautoencoder"]
cols = [C_RED, C_BLUE, C_GREEN, C_PURPLE]
w = 0.26
for i, (d, n_) in enumerate(zip(dets, names)):
    r_ = am["results"][d]
    ax.bar(np.arange(3) + (i - 1.5) * w, [r_["recall_spike"], r_["recall_shift"],
                                          r_["recall_dropout"]],
           width=w, color=cols[i], label=n_.replace("\n", " "))
ax.set_xticks(range(3))
ax.set_xticklabels(["Spike", "Level shift", "Dropout"])
ax.set_ylabel("Recall")
ax.set_title("(b) Recall by anomaly type")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_ylim(0, 1.08)
fig.savefig(f"{FIG}/fig6_anomaly_results.pdf")
plt.close(fig)
print("fig6 done")

# ================================================================ Fig 7: anomaly case study
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), constrained_layout=True)
ax = axes[0]
cs_start, cs_end = "2010-11-01", "2010-11-08"
seg = h["global_active_power"].loc[cs_start:cs_end]
zz = sc.set_index("timestamp")["z"].loc[cs_start:cs_end]
ax.plot(seg.index, seg.values, color=C_BLACK, lw=1.0, alpha=0.8, label="Actual")
an_hours = zz[zz > 3]
ax.scatter(an_hours.index, seg.reindex(an_hours.index), color=C_RED, s=34, zorder=5,
           marker="o", facecolors="none", linewidths=1.8, label="Flagged (|z| > 3)")
ax.axhline(0, color=C_GRAY, lw=0.6)
ax.set_ylabel("Active power (kW)")
ax.set_xlabel("Date")
ax.set_title("(a) Detected consumption anomalies (Nov 1–7, 2010)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

ax = axes[1]
inj_start, inj_end = "2010-08-02", "2010-08-09"
seg2 = s.loc[inj_start:inj_end]
zz2 = pd.Series(z_sc, index=te_cor.index).loc[inj_start:inj_end]
lab2 = pd.Series(lab_te, index=te_cor.index).loc[inj_start:inj_end]
ax.plot(seg2.index, seg2.values, color=C_BLACK, lw=1.0, alpha=0.8, label="Series (with injections)")
real = h["global_active_power"].loc[inj_start:inj_end]
ax.plot(real.index, real.values, color=C_GRAY, lw=0.8, ls=":", label="Original (clean)")
inj_h = lab2[lab2 == 1].index
ax.scatter(inj_h, seg2.reindex(inj_h), color=C_RED, s=30, zorder=5, marker="x", linewidths=1.8,
           label="Injected anomaly")
ax2 = ax.twinx()
ax2.plot(zz2.index, zz2.values, color=C_PURPLE, lw=1.1, alpha=0.9)
ax2.axhline(3, color=C_PURPLE, ls="--", lw=1)
ax2.set_ylabel("Anomaly score |z|", color=C_PURPLE)
ax2.tick_params(axis="y", labelcolor=C_PURPLE)
ax2.spines["right"].set_visible(True)
ax.set_ylabel("Active power (kW)")
ax.set_xlabel("Date")
ax.set_title("(b) Synthetic anomalies vs detector score (Aug 2010)")
ax.legend(fontsize=7.5, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.savefig(f"{FIG}/fig7_anomaly_casestudy.pdf")
plt.close(fig)
print("fig7 done")

# figure for the website hero + repo: PNG copies of fig1/fig3/fig6
for f in ["fig1_dataset_overview", "fig3_model_comparison", "fig6_anomaly_results", "fig4_forecast_quality"]:
    import subprocess
print("all figures done")

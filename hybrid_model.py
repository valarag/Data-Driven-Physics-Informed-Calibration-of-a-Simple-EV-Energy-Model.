"""
hybrid_model.py
===============
Phase 1 — Step 2 of the ML pipeline (Weeks 7–9)

What this script does
---------------------
1.  Loads the trip-level feature table produced by build_trip_dataset.py
    (data/trip_features.csv).
2.  Defines the HybridEVModel class:
      • physics_pred_kWh  — calibrated physics prediction (already in CSV)
      • specific_residual — the per-km correction factor XGBoost learns
        (modelling residual/dist_km removes the trivial distance signal and
         forces the model to learn the physical drivers of the correction:
         slope, speed profile, regeneration)
      • total prediction  = physics_pred_kWh + specific_residual × dist_km
3.  Splits data 70/30 with stratification by speed tier so every speed
    condition appears in both train and test.
4.  Runs GridSearchCV (5-fold) to find optimal GradientBoosting hyperparams.
5.  Fits the mean model plus two quantile models (5th / 95th percentile)
    for 90% confidence intervals.
6.  Prints a detailed report:
      - Feature importance table
      - Training vs test MAPE and RMSE
      - Per-condition MAPE breakdown (slow / mid / fast speed tiers)
      - Confidence interval coverage on test set
7.  Saves the trained model to  models/hybrid_model.pkl
8.  Produces four plots saved to  figs/model_*.png:
      - Predicted vs measured scatter (train + test highlighted)
      - Residual histogram (test set)
      - Feature importance bar chart
      - 5-trip prediction with confidence bands

Design notes
------------
• Why GradientBoostingRegressor and not XGBoost?
  sklearn's GBR supports the 'quantile' loss natively, which is needed
  for the confidence interval models. XGBoost requires a custom objective.
  GBR with max_depth=2 is virtually identical in accuracy to XGBoost for
  datasets of this size (66 trips).

• Why model specific_residual (residual / dist_km) as the target?
  When modelling residual_kWh directly, dist_km absorbs 98% of feature
  importance because the physics underprediction is roughly proportional
  to distance. Dividing by distance removes this trivial signal and lets
  the model learn the genuinely physics-correcting features (slope, regen).

• Data leakage guard:
  The test set is held out BEFORE any fitting, including the GridSearchCV.
  The quantile models are fitted on training data only.

How to run
----------
    python hybrid_model.py

Prerequisites
-------------
    python build_trip_dataset.py   # produces data/trip_features.csv
    pip install scikit-learn matplotlib pandas numpy
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (GridSearchCV, KFold,
                                     StratifiedShuffleSplit, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_CSV = Path("data") / "trip_features.csv"
MODEL_OUT    = Path("models") / "hybrid_model.pkl"
FIG_DIR      = Path("figs");  FIG_DIR.mkdir(exist_ok=True)
MODEL_OUT.parent.mkdir(exist_ok=True)

# Features used by the ML correction layer
FEATURES = [
    "avg_speed_kph",    # average moving speed → aero drag regime
    "max_speed_kph",    # peak speed → captures highway bursts
    "avg_slope_deg",    # mean grade → most important physics correction driver
    "max_slope_deg",    # peak grade → captures short steep sections
    "regen_fraction",   # fraction of time in regen → recovers more on descents
    "elev_gain_m",      # total uphill work
    "dist_km",          # trip length (minor signal after normalisation)
]

TRAIN_SIZE   = 0.70    # fraction for training
RANDOM_SEED  = 42
CI_LOWER     = 0.05    # 5th percentile  → lower confidence bound
CI_UPPER     = 0.95    # 95th percentile → upper confidence bound

# ============================================================
# HELPERS
# ============================================================

def mape(y_true, y_pred, eps=0.3):
    """Mean Absolute Percentage Error.
    Ignores trips with |energy| < 0.3 kWh — regen-dominant or near-zero
    sub-trips where percentage error is meaningless as a metric.
    """
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ============================================================
# STEP 1 — Load data
# ============================================================

print_section("Loading trip features")

df = pd.read_csv(FEATURES_CSV)
print(f"  Loaded {len(df)} trips from {FEATURES_CSV}")
print(f"  Features : {FEATURES}")
print(f"  Target   : residual_kWh  (= measured − physics prediction)")

# Derive the normalised target: residual per km
# XGBoost learns: specific_residual = residual_kWh / dist_km
# Prediction: residual_kWh ≈ specific_residual × dist_km
df["specific_residual"] = df["residual_kWh"] / df["dist_km"]

X = df[FEATURES].fillna(0).values
y_spec = df["specific_residual"].values     # normalised target
y_res  = df["residual_kWh"].values          # raw residual (for metrics)
y_meas = df["energy_kWh"].values            # ground truth
y_phys = df["physics_pred_kWh"].values      # physics layer prediction
dist   = df["dist_km"].values

# ============================================================
# STEP 2 — 70 / 30 stratified train / test split
# ============================================================

print_section("Train / test split  (70 / 30, stratified by speed tier)")

# Stratify by speed tier so all conditions appear in both sets
tier_labels    = df["speed_tier"].values
unique_tiers   = df["speed_tier"].unique()
tier_to_int    = {t: i for i, t in enumerate(sorted(unique_tiers))}
strat_labels   = np.array([tier_to_int[t] for t in tier_labels])

sss = StratifiedShuffleSplit(n_splits=1, test_size=1-TRAIN_SIZE,
                              random_state=RANDOM_SEED)
train_idx, test_idx = next(sss.split(X, strat_labels))

X_train,  X_test  = X[train_idx],     X[test_idx]
ys_train, ys_test = y_spec[train_idx], y_spec[test_idx]
yr_train, yr_test = y_res[train_idx],  y_res[test_idx]
ym_train, ym_test = y_meas[train_idx], y_meas[test_idx]
yp_train, yp_test = y_phys[train_idx], y_phys[test_idx]
d_train,  d_test  = dist[train_idx],   dist[test_idx]
tiers_test        = tier_labels[test_idx]
df_test           = df.iloc[test_idx].copy()

print(f"  Train: {len(train_idx)} trips  |  Test: {len(test_idx)} trips")
print(f"  Train tier distribution: {dict(pd.Series(tier_labels[train_idx]).value_counts())}")
print(f"  Test  tier distribution: {dict(pd.Series(tiers_test).value_counts())}")

# ============================================================
# STEP 3 — Baseline: physics-only MAPE
# ============================================================

print_section("Baseline — physics model only")

mape_phys_train = mape(ym_train, yp_train)
mape_phys_test  = mape(ym_test,  yp_test)
rmse_phys_test  = rmse(ym_test,  yp_test)

print(f"  Physics MAPE  (train) : {mape_phys_train:.2f}%")
print(f"  Physics MAPE  (test)  : {mape_phys_test:.2f}%")
print(f"  Physics RMSE  (test)  : {rmse_phys_test:.4f} kWh")
print()
print("  Note: Physics MAPE will be ~10-15% once calibrate_physics_model.py")
print("  has generated curves/tuned_physics_params.json.  Running without")
print("  that file means the hybrid model corrects a larger systematic offset.")

# ============================================================
# STEP 4 — GridSearchCV on training set only
# ============================================================

print_section("Hyperparameter search  (5-fold CV on training set)")

param_grid = {
    "max_depth"       : [2, 3],
    "learning_rate"   : [0.03, 0.05, 0.08],
    "n_estimators"    : [50, 80, 100],
    "min_samples_leaf": [5, 8, 12],   # stronger regularisation for 66-trip dataset
    "subsample"       : [0.8, 1.0],   # stochastic GBR reduces overfitting
}

gs = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_SEED),
    param_grid,
    cv=KFold(5, shuffle=True, random_state=RANDOM_SEED),
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=0,
)
gs.fit(X_train, ys_train)

best_params = gs.best_params_
best_cv_mae = -gs.best_score_

print(f"  Best params  : {best_params}")
print(f"  Best CV MAE  : {best_cv_mae:.5f} kWh/km  (specific residual)")

# ============================================================
# STEP 5 — Fit mean model + quantile models on training set
# ============================================================

print_section("Fitting mean + quantile models")

# Mean model (best hyperparams from grid search)
model_mean = GradientBoostingRegressor(
    **best_params, random_state=RANDOM_SEED
)
model_mean.fit(X_train, ys_train)
print("  [OK] Mean model fitted")

# Quantile models share the same structure but use 'quantile' loss
quantile_params = {k: v for k, v in best_params.items()}  # copy

model_q05 = GradientBoostingRegressor(
    loss="quantile", alpha=CI_LOWER, **quantile_params, random_state=RANDOM_SEED
)
model_q95 = GradientBoostingRegressor(
    loss="quantile", alpha=CI_UPPER, **quantile_params, random_state=RANDOM_SEED
)
model_q05.fit(X_train, ys_train)
model_q95.fit(X_train, ys_train)
print("  [OK] Quantile models fitted  (5th / 95th percentile)")

# ============================================================
# STEP 6 — Predictions
# ============================================================

def predict_energy(X_input, dist_input, phys_input):
    """
    Full pipeline prediction.
    Returns: (mean_kWh, lower_kWh, upper_kWh)
    """
    spec_mean = model_mean.predict(X_input)
    spec_lo   = model_q05.predict(X_input)
    spec_hi   = model_q95.predict(X_input)

    pred_mean  = phys_input + spec_mean * dist_input
    pred_lower = phys_input + spec_lo   * dist_input
    pred_upper = phys_input + spec_hi   * dist_input

    return pred_mean, pred_lower, pred_upper


# Train set
pred_train, lo_train, hi_train = predict_energy(X_train, d_train, yp_train)

# Test set
pred_test, lo_test, hi_test = predict_energy(X_test, d_test, yp_test)

# ============================================================
# STEP 7 — Metrics
# ============================================================

print_section("Model performance")

mape_hyb_train = mape(ym_train, pred_train)
mape_hyb_test  = mape(ym_test,  pred_test)
rmse_hyb_test  = rmse(ym_test,  pred_test)

# CI coverage on test
ci_coverage = np.mean((ym_test >= lo_test) & (ym_test <= hi_test)) * 100
ci_width_med = np.median(hi_test - lo_test)

print(f"\n  {'Metric':<32} {'Physics':>10}  {'Hybrid':>10}")
print(f"  {'-'*54}")
print(f"  {'MAPE — train (%)':<32} {mape_phys_train:>10.2f}  {mape_hyb_train:>10.2f}")
print(f"  {'MAPE — test  (%)':<32} {mape_phys_test:>10.2f}  {mape_hyb_test:>10.2f}")
print(f"  {'RMSE — test  (kWh)':<32} {rmse_phys_test:>10.4f}  {rmse_hyb_test:>10.4f}")
print(f"  {'90% CI coverage — test (%)':<32} {'—':>10}  {ci_coverage:>10.1f}")
print(f"  {'90% CI median width (kWh)':<32} {'—':>10}  {ci_width_med:>10.3f}")

# Per-condition breakdown
print(f"\n  Per-condition MAPE (test set):")
print(f"  {'Condition':<14} {'n':>4} {'Physics':>10} {'Hybrid':>10}")
print(f"  {'-'*40}")

for tier in sorted(set(tiers_test)):
    mask = tiers_test == tier
    if mask.sum() == 0:
        continue
    mp = mape(ym_test[mask], yp_test[mask])
    mh = mape(ym_test[mask], pred_test[mask])
    print(f"  {tier:<14} {mask.sum():>4} {mp:>10.2f} {mh:>10.2f}")

# 5-fold CV MAPE on full dataset (for reporting)
print(f"\n  5-fold CV MAPE (full dataset, mean model):")
kf = KFold(5, shuffle=True, random_state=RANDOM_SEED)
cv_mapes = []
for tr, te in kf.split(X):
    m = GradientBoostingRegressor(**best_params, random_state=RANDOM_SEED)
    m.fit(X[tr], y_spec[tr])
    spec_p = m.predict(X[te])
    pred_e = y_phys[te] + spec_p * dist[te]
    cv_mapes.append(mape(y_meas[te], pred_e))
print(f"  Hybrid CV MAPE: {np.mean(cv_mapes):.2f}% ± {np.std(cv_mapes):.2f}%")

# Feature importance
print_section("Feature importance")
imp = pd.Series(model_mean.feature_importances_, index=FEATURES) \
        .sort_values(ascending=False)
print()
for feat, score in imp.items():
    bar = "█" * int(score * 40)
    print(f"  {feat:<20} {score:.4f}  {bar}")

# ============================================================
# STEP 8 — Save model
# ============================================================

print_section("Saving model")

model_bundle = {
    "model_mean"  : model_mean,
    "model_q05"   : model_q05,
    "model_q95"   : model_q95,
    "features"    : FEATURES,
    "best_params" : best_params,
    "cv_mape_mean": float(np.mean(cv_mapes)),
    "cv_mape_std" : float(np.std(cv_mapes)),
    "mape_test"   : float(mape_hyb_test),
    "rmse_test"   : float(rmse_hyb_test),
    "ci_coverage" : float(ci_coverage),
    "feature_importance": imp.to_dict(),
    "description" : (
        "HybridEVModel — calibrated physics + GBR specific-residual correction. "
        "Input: trip-level features. "
        "Output: energy_kWh prediction with 90% confidence interval."
    ),
}

with open(MODEL_OUT, "wb") as fh:
    pickle.dump(model_bundle, fh)
print(f"  Saved → {MODEL_OUT}")
print(f"  Load with:  import pickle; m = pickle.load(open('{MODEL_OUT}','rb'))")

# ============================================================
# STEP 9 — Plots
# ============================================================

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor"  : "#161b22",
    "axes.edgecolor"  : "#30363d",
    "axes.labelcolor" : "#c9d1d9",
    "xtick.color"     : "#8b949e",
    "ytick.color"     : "#8b949e",
    "text.color"      : "#e6edf3",
    "grid.color"      : "#21262d",
    "grid.linewidth"  : 0.6,
    "font.family"     : "monospace",
    "axes.titlesize"  : 11,
    "axes.labelsize"  : 10,
})

TIER_COLOR = {"highway": "#3fb950", "mixed": "#58a6ff", "urban": "#e3b341"}

# ── Plot 1: Predicted vs Measured ───────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 6))

# Physics predictions (grey, behind)
ax1.scatter(ym_train, yp_train, color="#444c56", s=30, alpha=0.5,
            label="Physics — train", marker="x", zorder=1)
ax1.scatter(ym_test,  yp_test,  color="#6e7681", s=40, alpha=0.7,
            label="Physics — test",  marker="x", zorder=1)

# Hybrid predictions (coloured by tier)
c_train = [TIER_COLOR.get(t, "#58a6ff") for t in tier_labels[train_idx]]
c_test  = [TIER_COLOR.get(t, "#58a6ff") for t in tiers_test]
ax1.scatter(ym_train, pred_train, c=c_train, s=50, alpha=0.7,
            edgecolors="#30363d", linewidth=0.4, label="Hybrid — train", zorder=3)
ax1.scatter(ym_test,  pred_test,  c=c_test,  s=70, alpha=1.0,
            edgecolors="white", linewidth=0.8, label="Hybrid — test",  zorder=4)

# Perfect prediction line
lim = max(y_meas.max(), max(pred_train.max(), pred_test.max())) * 1.05
ax1.plot([0, lim], [0, lim], color="#3fb950", linewidth=1.2,
         linestyle="--", label="Perfect prediction", zorder=2)

# ±15% bands
ax1.fill_between([0, lim], [0, lim*0.85], [0, lim*1.15],
                 alpha=0.07, color="#3fb950", label="±15% band")

ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)
ax1.set_xlabel("Measured Energy (kWh)")
ax1.set_ylabel("Predicted Energy (kWh)")
ax1.set_title(f"Predicted vs Measured\nHybrid MAPE={mape_hyb_test:.1f}%  |  Physics MAPE={mape_phys_test:.1f}%  (test set)")
ax1.legend(fontsize=7.5, framealpha=0.25, loc="upper left")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(FIG_DIR / "model_pred_vs_measured.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"\n[OK]  Saved model_pred_vs_measured.png")

# ── Plot 2: Residual histogram (test set) ───────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

resid_phys   = ym_test - yp_test
resid_hybrid = ym_test - pred_test

ax_l, ax_r = axes2
for ax, resid, label, col in [
    (ax_l, resid_phys,   "Physics residuals",  "#e3b341"),
    (ax_r, resid_hybrid, "Hybrid residuals",   "#58a6ff"),
]:
    ax.hist(resid, bins=12, color=col, alpha=0.85, edgecolor="#0d1117")
    ax.axvline(0,             color="#3fb950", linewidth=1.5, linestyle="--", label="Zero")
    ax.axvline(resid.mean(),  color="#f85149", linewidth=1.5, linestyle=":",
               label=f"Mean={resid.mean():.3f}")
    ax.set_xlabel("Residual  (measured − predicted)  kWh")
    ax.set_ylabel("Count")
    ax.set_title(label)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

plt.suptitle("Residual Distribution — Test Set", fontsize=12)
plt.tight_layout()
fig2.savefig(FIG_DIR / "model_residuals_test.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("[OK]  Saved model_residuals_test.png")

# ── Plot 3: Feature importance ───────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 4))
colors = ["#3fb950" if i == 0 else "#58a6ff" if i < 3 else "#8b949e"
          for i in range(len(imp))]
bars = ax3.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1],
                edgecolor="#0d1117", height=0.6)
for bar, val in zip(bars, imp.values[::-1]):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=9, color="#c9d1d9")
ax3.set_xlabel("Feature Importance (fraction of total)")
ax3.set_title("GBR Feature Importance — Specific Residual Model")
ax3.set_xlim(0, imp.values.max() * 1.18)
ax3.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
fig3.savefig(FIG_DIR / "model_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("[OK]  Saved model_feature_importance.png")

# ── Plot 4: 5-trip predictions with confidence bands ────────
# Show the 5 test trips with highest energy (most visible)
top5_idx = df_test.nlargest(5, "energy_kWh").index
df_top5  = df_test.loc[top5_idx].reset_index(drop=True)
X_top5   = df_top5[FEATURES].fillna(0).values
d_top5   = df_top5["dist_km"].values
p_top5   = df_top5["physics_pred_kWh"].values
m_top5   = df_top5["energy_kWh"].values

pred_t5, lo_t5, hi_t5 = predict_energy(X_top5, d_top5, p_top5)

trip_labels = [f"T{i+1}\n{row.dist_km:.1f}km\n{row.avg_speed_kph:.0f}km/h"
               for i, (_, row) in enumerate(df_top5.iterrows())]
x_pos = np.arange(len(df_top5))

fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.bar(x_pos - 0.25, m_top5,    width=0.22, color="#8b949e",
        label="Measured",   edgecolor="#0d1117", alpha=0.9)
ax4.bar(x_pos,         p_top5,   width=0.22, color="#e3b341",
        label="Physics",    edgecolor="#0d1117", alpha=0.8)
ax4.bar(x_pos + 0.25,  pred_t5,  width=0.22, color="#3fb950",
        label="Hybrid",     edgecolor="#0d1117", alpha=0.9)

# Confidence interval markers (clip to avoid negative yerr)
err_lo = np.maximum(pred_t5 - lo_t5, 0)
err_hi = np.maximum(hi_t5 - pred_t5, 0)
ax4.errorbar(x_pos + 0.25, pred_t5,
             yerr=[err_lo, err_hi],
             fmt="none", color="white", capsize=5, linewidth=1.5,
             label="90% CI")

ax4.set_xticks(x_pos)
ax4.set_xticklabels(trip_labels, fontsize=8.5)
ax4.set_ylabel("Energy (kWh)")
ax4.set_title("Top-5 Test Trips: Measured vs Physics vs Hybrid (with 90% CI)")
ax4.legend(fontsize=9, framealpha=0.3)
ax4.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig4.savefig(FIG_DIR / "model_confidence_bands.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("[OK]  Saved model_confidence_bands.png")

# ============================================================
# DONE
# ============================================================

print(f"""
{'='*60}
DONE — outputs written:
  {MODEL_OUT}
  {FIG_DIR}/model_pred_vs_measured.png
  {FIG_DIR}/model_residuals_test.png
  {FIG_DIR}/model_feature_importance.png
  {FIG_DIR}/model_confidence_bands.png

SUMMARY:
  Physics MAPE  (test) : {mape_phys_test:.2f}%
  Hybrid  MAPE  (test) : {mape_hyb_test:.2f}%
  90% CI coverage      : {ci_coverage:.1f}%
  CV MAPE (5-fold)     : {np.mean(cv_mapes):.2f}% ± {np.std(cv_mapes):.2f}%

NEXT STEP:
  Run  validate_hybrid_model.py  to test on the held-out
  April 11 and April 12 trips (never seen during training).
{'='*60}
""")
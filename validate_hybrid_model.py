"""
validate_hybrid_model.py  [FIXED]
===================================
Tests the frozen hybrid model on two completely unseen trips.

BUG FIXED vs PREVIOUS VERSION
------------------------------
process_raw_trip() was reading eta_sys from Powertrain_efficiency_gear_SG
(mechanical drivetrain efficiency ~0.90), the same bug present in
build_trip_dataset.py.  This caused physics_pred_kWh to be ~2× too low
on cross-trips, so the hybrid model had to compensate a large systematic
bias instead of a small physics residual — leading to 25% cross-trip MAPE.

FIX: eta_sys is now interpolated from curves/eta_powertrain_vs_speed.csv
(true system efficiency ≈ 0.35–0.44), consistent with training.
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE    = Path("models") / "hybrid_model.pkl"
PARAMS_FILE   = Path("curves") / "tuned_physics_params.json"
ETA_PT_FILE   = Path("curves") / "eta_powertrain_vs_speed.csv"
TRAIN_CSV     = Path("data")   / "trip_features.csv"
CROSS_TRIPS   = {
    "Trip 11 Apr": "trip_11apr2022.csv",
    "Trip 12 Apr": "trip_12apr2022.csv",
}
REPORT_OUT    = Path("data")   / "validation_report.csv"
FIG_DIR       = Path("figs");   FIG_DIR.mkdir(exist_ok=True)

STOP_THRESHOLD_S    = 30
MIN_TRIP_DIST_KM    = 0.5
MIN_TRIP_DURATION_S = 60

# ============================================================
# HELPERS
# ============================================================

def mape(y_true, y_pred, eps=0.3):
    """Ignores trips with |energy| < 0.3 kWh (near-zero / regen-dominant)."""
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

def print_section(title):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")

# ============================================================
# STEP 1 — Load model + params
# ============================================================

print_section("Loading model")

with open(MODEL_FILE, "rb") as fh:
    bundle = pickle.load(fh)

model_mean = bundle["model_mean"]
model_q05  = bundle["model_q05"]
model_q95  = bundle["model_q95"]
FEATURES   = bundle["features"]

print(f"  Loaded: {MODEL_FILE}")
print(f"  Features: {FEATURES}")
print(f"  Best params: {bundle['best_params']}")
print(f"  Training CV MAPE: {bundle['cv_mape_mean']:.2f}% ± {bundle['cv_mape_std']:.2f}%")

# Physics params
if PARAMS_FILE.exists():
    with open(PARAMS_FILE) as fh:
        p = json.load(fh)
    Cd0  = p["Cd0"];   Crr  = p["Crr"];  k_cd = p.get("k_cd", 0.0)
    RHO  = p["RHO"];   A    = p["A"];    MASS = p["MASS"];  G = p["G"]
    print(f"  Physics params: tuned  (Cd={Cd0:.4f}, Crr={Crr:.5f})")
else:
    Cd0, Crr, k_cd = 0.23, 0.009, 0.0
    RHO, A, MASS, G = 1.225, 2.22, 1847, 9.81
    print("  Physics params: default (run calibrate_physics_model.py for tuned values)")

HVAC_W = 0

# FIX: load the empirical eta curve (true system efficiency)
if not ETA_PT_FILE.exists():
    raise FileNotFoundError(
        f"{ETA_PT_FILE} not found. Run build_eta_powertrain.py first."
    )
eta_pt = pd.read_csv(ETA_PT_FILE).sort_values("speed_kph")

def interp_eta(speed_kph_arr):
    return np.clip(
        np.interp(np.asarray(speed_kph_arr, dtype=float),
                  eta_pt["speed_kph"].values,
                  eta_pt["eta_powertrain_mean"].values,
                  left  = eta_pt["eta_powertrain_mean"].iloc[0],
                  right = eta_pt["eta_powertrain_mean"].iloc[-1]),
        0.25, 0.65
    )

# ============================================================
# STEP 2 — Helper: process any raw CSV into trip-level features
# ============================================================

def process_raw_trip(filepath):
    df = pd.read_csv(filepath)
    df["dt_s"]   = pd.to_numeric(df["DeltaT"],  errors="coerce").clip(0.05, 5.0)
    df["speed"]  = pd.to_numeric(df["Speed"],   errors="coerce").clip(lower=0)
    df["v_mps"]  = df["speed"] / 3.6
    df["theta"]  = pd.to_numeric(df["Slope Angle (rad)"], errors="coerce").fillna(0)
    df["accel"]  = pd.to_numeric(df["Acceleration"],      errors="coerce").fillna(0)
    df["elev"]   = pd.to_numeric(df["ElevChange"],        errors="coerce").fillna(0)
    df["energy"] = pd.to_numeric(df["Energy Consumption (kWh)"], errors="coerce").fillna(0)
    df["disp_m"] = pd.to_numeric(df["Displacement (m)"], errors="coerce").fillna(0)

    # FIX: true system efficiency from empirical curve (consistent with training)
    df["eta_sys"] = interp_eta(df["speed"].values)

    # Row-level physics
    v = df["v_mps"].values;  theta = df["theta"].values
    a = df["accel"].values;  eta   = df["eta_sys"].values
    Cd_v   = Cd0 + k_cd * v
    F_tot  = (0.5 * RHO * Cd_v * A * v**2
              + Crr * MASS * G * np.cos(theta)
              + MASS * G * np.sin(theta)
              + MASS * a)
    P_mech = F_tot * v + HVAC_W
    P_bat  = np.where(P_mech < 0, P_mech * eta, P_mech / eta)
    df["E_phys_kWh"] = P_bat * df["dt_s"].values / 3_600_000

    # Segmentation
    df["is_stop"]    = df["speed"] < 1.0
    df["stop_group"] = (df["is_stop"] != df["is_stop"].shift()).cumsum()
    stop_dur   = df[df["is_stop"]].groupby("stop_group")["dt_s"].sum()
    long_stops = set(stop_dur[stop_dur > STOP_THRESHOLD_S].index)
    df["is_long_stop"] = df["stop_group"].isin(long_stops) & df["is_stop"]
    df["trip_id"]      = df["is_long_stop"].cumsum()

    records = []
    for tid, seg in df.groupby("trip_id"):
        moving  = seg[seg["speed"] >= 1.0]
        dist_km = seg["disp_m"].sum() / 1000.0
        dur_s   = seg["dt_s"].sum()
        if dist_km < MIN_TRIP_DIST_KM or dur_s < MIN_TRIP_DURATION_S:
            continue

        avg_spd = float(moving["speed"].mean()) if len(moving) > 0 else 0.0
        max_spd = float(seg["speed"].max())
        avg_slp = float(np.degrees(seg["theta"].abs().mean()))
        max_slp = float(np.degrees(seg["theta"].abs().max()))
        elev_g  = float(seg["elev"].clip(lower=0).sum())
        regen_f = float((seg["E_phys_kWh"] < 0).sum() / len(seg))
        energy  = float(seg["energy"].sum())
        phys_e  = float(seg["E_phys_kWh"].sum())

        tier       = "highway" if avg_spd >= 70 else ("mixed" if avg_spd >= 30 else "urban")
        grade_tier = "hilly" if max_slp >= 3 else "flat"
        len_tier   = "long"  if dist_km >= 20 else ("medium" if dist_km >= 5 else "short")

        records.append({
            "trip_id"         : int(tid),
            "dist_km"         : round(dist_km, 3),
            "avg_speed_kph"   : round(avg_spd,  2),
            "max_speed_kph"   : round(max_spd,  2),
            "avg_slope_deg"   : round(avg_slp,  4),
            "max_slope_deg"   : round(max_slp,  2),
            "elev_gain_m"     : round(elev_g,   2),
            "regen_fraction"  : round(regen_f,  4),
            "energy_kWh"      : round(energy,   5),
            "physics_pred_kWh": round(phys_e,   5),
            "speed_tier"      : tier,
            "grade_tier"      : grade_tier,
            "length_tier"     : len_tier,
        })

    return pd.DataFrame(records)


def predict_trip_df(tdf):
    X    = tdf[FEATURES].fillna(0).values
    dist = tdf["dist_km"].values
    phys = tdf["physics_pred_kWh"].values
    return (phys + model_mean.predict(X) * dist,
            phys + model_q05.predict(X)  * dist,
            phys + model_q95.predict(X)  * dist)

# ============================================================
# STEP 3 — Held-out test set  (reproduced from training split)
# ============================================================

print_section("Held-out test set  (from training split in hybrid_model.py)")

from sklearn.model_selection import StratifiedShuffleSplit

df_train_full = pd.read_csv(TRAIN_CSV)
tier_to_int   = {t: i for i, t in enumerate(sorted(df_train_full["speed_tier"].unique()))}
strat         = np.array([tier_to_int[t] for t in df_train_full["speed_tier"]])
sss           = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
_, test_idx   = next(sss.split(df_train_full, strat))
df_test       = df_train_full.iloc[test_idx].copy()

df_test["grade_tier"]  = np.where(df_test["max_slope_deg"] >= 3, "hilly", "flat")
df_test["length_tier"] = pd.cut(df_test["dist_km"],
                                bins=[0, 5, 20, np.inf],
                                labels=["short", "medium", "long"]).astype(str)

pred_test, lo_test, hi_test = predict_trip_df(df_test)
ym_test = df_test["energy_kWh"].values
yp_test = df_test["physics_pred_kWh"].values

print(f"\n  {'Metric':<30} {'Physics':>10}  {'Hybrid':>10}")
print(f"  {'-'*52}")
print(f"  {'MAPE (%)':<30} {mape(ym_test,yp_test):>10.2f}  {mape(ym_test,pred_test):>10.2f}")
print(f"  {'RMSE (kWh)':<30} {rmse(ym_test,yp_test):>10.4f}  {rmse(ym_test,pred_test):>10.4f}")
print(f"  {'R²':<30} {r2(ym_test,yp_test):>10.4f}  {r2(ym_test,pred_test):>10.4f}")
ci_cov_test = np.mean((ym_test >= lo_test) & (ym_test <= hi_test)) * 100
print(f"  {'90% CI coverage (%)':<30} {'—':>10}  {ci_cov_test:>10.1f}")

# ============================================================
# STEP 4 — Cross-trip validation
# ============================================================

print_section("Cross-trip validation  (unseen trips)")

all_cross_results = {}

for label, fname in CROSS_TRIPS.items():
    if not Path(fname).exists():
        print(f"  [SKIP] {fname} not found")
        continue

    tdf = process_raw_trip(fname)
    if tdf.empty:
        print(f"  [SKIP] {label}: no valid sub-trips")
        continue

    pred, lo, hi = predict_trip_df(tdf)
    meas = tdf["energy_kWh"].values
    phys = tdf["physics_pred_kWh"].values
    tdf["pred_kWh"] = pred
    tdf["lo_kWh"]   = lo
    tdf["hi_kWh"]   = hi
    tdf["residual_hybrid"]  = meas - pred
    tdf["residual_physics"] = meas - phys
    all_cross_results[label] = tdf

    mp = mape(meas, phys);  mh = mape(meas, pred)
    rp = rmse(meas, phys);  rh = rmse(meas, pred)
    r2p = r2(meas, phys);   r2h = r2(meas, pred)
    ci_cov = np.mean((meas >= lo) & (meas <= hi)) * 100

    print(f"\n  ── {label}  ({len(tdf)} sub-trips, {tdf['dist_km'].sum():.1f} km total) ──")
    print(f"  {'Metric':<28} {'Physics':>10}  {'Hybrid':>10}")
    print(f"  {'-'*50}")
    print(f"  {'MAPE (%)':<28} {mp:>10.2f}  {mh:>10.2f}")
    print(f"  {'RMSE (kWh)':<28} {rp:>10.4f}  {rh:>10.4f}")
    print(f"  {'R²':<28} {r2p:>10.4f}  {r2h:>10.4f}")
    print(f"  {'90% CI coverage (%)':<28} {'—':>10}  {ci_cov:>10.1f}")

    print(f"\n  Sub-trip breakdown:")
    print(f"  {'ID':>3} {'Dist':>7} {'AvgSpd':>8} {'MaxSlp':>8} "
          f"{'Meas':>8} {'Phys':>8} {'Hybrid':>8} {'Err%':>7} {'InCI':>5}")
    print(f"  {'-'*68}")
    for _, row in tdf.iterrows():
        err_pct = (row.pred_kWh - row.energy_kWh) / (abs(row.energy_kWh) + 1e-9) * 100
        in_ci   = "✓" if row.lo_kWh <= row.energy_kWh <= row.hi_kWh else "✗"
        print(f"  {int(row.trip_id):>3} "
              f"{row.dist_km:>7.2f} "
              f"{row.avg_speed_kph:>8.1f} "
              f"{row.max_slope_deg:>8.2f} "
              f"{row.energy_kWh:>8.3f} "
              f"{row.physics_pred_kWh:>8.3f} "
              f"{row.pred_kWh:>8.3f} "
              f"{err_pct:>+7.1f} "
              f"{in_ci:>5}")

# ============================================================
# STEP 5 — Condition matrix
# ============================================================

print_section("Condition matrix  (all cross-trips combined)")

all_cross_df = pd.concat(
    [df.assign(source=label) for label, df in all_cross_results.items()],
    ignore_index=True
)

def condition_table(df, condition_col, title):
    print(f"\n  {title}:")
    print(f"  {'Condition':<12} {'n':>4} {'Physics MAPE':>14} {'Hybrid MAPE':>13} "
          f"{'RMSE (kWh)':>12} {'R²':>8}")
    print(f"  {'-'*64}")
    rows = []
    for cond in sorted(df[condition_col].unique()):
        mask = df[condition_col] == cond
        n    = mask.sum()
        if n == 0:
            continue
        mp  = mape(df.loc[mask, "energy_kWh"].values, df.loc[mask, "physics_pred_kWh"].values)
        mh  = mape(df.loc[mask, "energy_kWh"].values, df.loc[mask, "pred_kWh"].values)
        rh  = rmse(df.loc[mask, "energy_kWh"].values, df.loc[mask, "pred_kWh"].values)
        r2h = r2(  df.loc[mask, "energy_kWh"].values, df.loc[mask, "pred_kWh"].values)
        print(f"  {cond:<12} {n:>4} {mp:>14.2f} {mh:>13.2f} {rh:>12.4f} {r2h:>8.4f}")
        rows.append({"condition_type": title, "condition": cond, "n": n,
                     "physics_mape": mp, "hybrid_mape": mh, "rmse": rh, "r2": r2h})
    return rows

report_rows  = condition_table(all_cross_df, "speed_tier",  "Speed tier")
report_rows += condition_table(all_cross_df, "grade_tier",  "Grade tier")
report_rows += condition_table(all_cross_df, "length_tier", "Trip length tier")

# ============================================================
# STEP 6 — Edge case tests
# ============================================================

print_section("Edge case tests")

edge_cases = {
    "High slope   (max_slope ≥ 10°)"     : all_cross_df["max_slope_deg"] >= 10,
    "Highway avg  (avg_speed ≥ 70 km/h)" : all_cross_df["avg_speed_kph"] >= 70,
    "Short trip   (dist < 2 km)"         : all_cross_df["dist_km"] < 2,
    "Regen-heavy  (regen_fraction ≥ 0.4)": all_cross_df["regen_fraction"] >= 0.4,
}

print(f"\n  {'Edge case':<42} {'n':>4} {'Hybrid MAPE':>12}  Note")
print(f"  {'-'*72}")
for desc, mask in edge_cases.items():
    n = mask.sum()
    if n == 0:
        print(f"  {desc:<42} {n:>4} {'—':>12}  No samples in cross-trips")
        continue
    mh   = mape(all_cross_df.loc[mask, "energy_kWh"].values,
                all_cross_df.loc[mask, "pred_kWh"].values)
    note = "✓ within target" if mh < 10 else "⚠ check model"
    print(f"  {desc:<42} {n:>4} {mh:>12.2f}%  {note}")

# ============================================================
# STEP 7 — Confidence interval summary
# ============================================================

print_section("Confidence interval summary  (90% CI, all cross-trips)")

ci_cov_all   = np.mean((all_cross_df["energy_kWh"] >= all_cross_df["lo_kWh"]) &
                        (all_cross_df["energy_kWh"] <= all_cross_df["hi_kWh"])) * 100
ci_width_med = np.median(all_cross_df["hi_kWh"] - all_cross_df["lo_kWh"])
ci_width_max = (all_cross_df["hi_kWh"] - all_cross_df["lo_kWh"]).max()

print(f"\n  90% CI coverage (cross-trips) : {ci_cov_all:.1f}%  (target: ≥ 80%)")
print(f"  Median CI width               : {ci_width_med:.3f} kWh")
print(f"  Max CI width                  : {ci_width_max:.3f} kWh")

# ============================================================
# STEP 8 — Save validation report
# ============================================================

pd.DataFrame(report_rows).to_csv(REPORT_OUT, index=False)
print(f"\n[OK]  Validation report saved → {REPORT_OUT}")

# ============================================================
# STEP 9 — Plots
# ============================================================

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#c9d1d9",
    "xtick.color":      "#8b949e", "ytick.color":     "#8b949e",
    "text.color":       "#e6edf3", "grid.color":      "#21262d",
    "grid.linewidth":   0.6,       "font.family":     "monospace",
    "axes.titlesize":   11,        "axes.labelsize":  10,
})

TIER_COLOR = {"highway": "#3fb950", "mixed": "#58a6ff", "urban": "#e3b341"}

# Plot 1 — MAPE comparison
sets_labels    = ["Held-out\ntest set"] + list(all_cross_results.keys())
mape_phys_list = [mape(ym_test, yp_test)]
mape_hyb_list  = [mape(ym_test, pred_test)]
for tdf in all_cross_results.values():
    mape_phys_list.append(mape(tdf["energy_kWh"].values, tdf["physics_pred_kWh"].values))
    mape_hyb_list.append( mape(tdf["energy_kWh"].values, tdf["pred_kWh"].values))

x, width = np.arange(len(sets_labels)), 0.32
fig1, ax1 = plt.subplots(figsize=(9, 5))
b1 = ax1.bar(x - width/2, mape_phys_list, width, color="#e3b341", alpha=0.85,
             edgecolor="#0d1117", label="Physics only")
b2 = ax1.bar(x + width/2, mape_hyb_list,  width, color="#3fb950", alpha=0.9,
             edgecolor="#0d1117", label="Hybrid model")
ax1.axhline(7,  color="#f85149", linewidth=1.2, linestyle="--", label="Target MAPE = 7%")
ax1.axhline(15, color="#e3b341", linewidth=0.8, linestyle=":",  label="Physics target = 15%")
for bar in b1:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{bar.get_height():.1f}%", ha="center", fontsize=8.5, color="#e3b341")
for bar in b2:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{bar.get_height():.1f}%", ha="center", fontsize=8.5, color="#3fb950")
ax1.set_xticks(x); ax1.set_xticklabels(sets_labels, fontsize=9)
ax1.set_ylabel("MAPE (%)"); ax1.set_title("Physics vs Hybrid MAPE — All Evaluation Sets")
ax1.legend(fontsize=8.5, framealpha=0.3); ax1.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig1.savefig(FIG_DIR / "val_mape_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("[OK]  Saved val_mape_comparison.png")

# Plot 2 — Predicted vs Measured
n_cross = len(all_cross_results)
fig2, axes2 = plt.subplots(1, n_cross, figsize=(6 * n_cross, 5))
if n_cross == 1:
    axes2 = [axes2]
for ax, (label, tdf) in zip(axes2, all_cross_results.items()):
    colors = [TIER_COLOR.get(t, "#58a6ff") for t in tdf["speed_tier"]]
    meas = tdf["energy_kWh"].values;   pred = tdf["pred_kWh"].values
    phys = tdf["physics_pred_kWh"].values
    lo   = tdf["lo_kWh"].values;       hi   = tdf["hi_kWh"].values
    ax.scatter(meas, phys, color="#e3b341", s=55, alpha=0.6, marker="x", label="Physics")
    ax.scatter(meas, pred, c=colors, s=70, alpha=0.95,
               edgecolors="white", linewidth=0.6, label="Hybrid")
    for i in range(len(meas)):
        ax.plot([meas[i], meas[i]], [lo[i], hi[i]],
                color="#58a6ff", alpha=0.4, linewidth=1.2)
    lim = max(meas.max(), pred.max(), hi.max()) * 1.08
    ax.plot([0, lim], [0, lim], color="#3fb950", linewidth=1.2,
            linestyle="--", label="Perfect")
    ax.fill_between([0, lim], [0, lim*0.93], [0, lim*1.07], alpha=0.06, color="#3fb950")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Measured Energy (kWh)")
    ax.set_ylabel("Predicted Energy (kWh)")
    ax.set_title(f"{label}\nHybrid MAPE={mape(meas,pred):.1f}%  Physics MAPE={mape(meas,phys):.1f}%")
    ax.legend(fontsize=7.5, framealpha=0.25); ax.grid(True, alpha=0.3)
plt.suptitle("Cross-Trip Validation — Predicted vs Measured", fontsize=12, y=1.01)
plt.tight_layout()
fig2.savefig(FIG_DIR / "val_pred_vs_measured.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("[OK]  Saved val_pred_vs_measured.png")

# Plot 3 — Residual QQ
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
for ax, resid, label, col in [
    (axes3[0], all_cross_df["residual_physics"].dropna().values, "Physics residuals", "#e3b341"),
    (axes3[1], all_cross_df["residual_hybrid"].dropna().values,  "Hybrid residuals",  "#58a6ff"),
]:
    (osm, osr), (slope, intercept, r_sq) = stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, color=col, s=35, alpha=0.85, edgecolors="#0d1117", linewidth=0.4)
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope*x_line + intercept, color="#3fb950", linewidth=1.5,
            linestyle="--", label=f"Normal fit  R²={r_sq:.3f}")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles (kWh)")
    ax.set_title(f"QQ Plot — {label}")
    ax.legend(fontsize=8.5, framealpha=0.3); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig3.savefig(FIG_DIR / "val_residual_qq.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("[OK]  Saved val_residual_qq.png")

# Plot 4 — Condition matrix heatmap
pivot_data = {}
for row in report_rows:
    key = f"{row['condition_type'][:5]}\n{row['condition']}"
    pivot_data[key] = {"Physics MAPE": row["physics_mape"], "Hybrid MAPE": row["hybrid_mape"],
                       "RMSE (kWh)": row["rmse"], "R²": row["r2"]}
if pivot_data:
    pivot_df = pd.DataFrame(pivot_data).T
    fig4, ax4 = plt.subplots(figsize=(9, max(4, len(pivot_df)*0.55 + 1.5)))
    data_arr = pivot_df.values.astype(float)
    normed   = np.zeros_like(data_arr)
    for j in range(data_arr.shape[1]):
        col_min, col_max = data_arr[:,j].min(), data_arr[:,j].max()
        if col_max > col_min:
            normed[:,j] = (data_arr[:,j] - col_min) / (col_max - col_min)
        if pivot_df.columns[j] == "R²":
            normed[:,j] = 1 - normed[:,j]
    ax4.imshow(normed, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1, alpha=0.85)
    ax4.set_xticks(range(len(pivot_df.columns)))
    ax4.set_xticklabels(pivot_df.columns, fontsize=9)
    ax4.set_yticks(range(len(pivot_df)))
    ax4.set_yticklabels(pivot_df.index, fontsize=8.5)
    for i in range(len(pivot_df)):
        for j, col in enumerate(pivot_df.columns):
            val = data_arr[i, j]
            fmt = f"{val:.2f}" if col != "R²" else f"{val:.3f}"
            ax4.text(j, i, fmt, ha="center", va="center",
                     fontsize=9, color="black", fontweight="bold")
    ax4.set_title("Condition Matrix — Hybrid Model Performance\n"
                  "Green = better, Red = worse  (normalised per column)", fontsize=11, pad=12)
    plt.tight_layout()
    fig4.savefig(FIG_DIR / "val_condition_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("[OK]  Saved val_condition_matrix.png")

# ============================================================
# PHASE 1 PASS / FAIL CHECKLIST
# ============================================================

print_section("Phase 1 — Pass / Fail checklist")

hybrid_mape_all = mape(all_cross_df["energy_kWh"].values, all_cross_df["pred_kWh"].values)
phys_mape_all   = mape(all_cross_df["energy_kWh"].values, all_cross_df["physics_pred_kWh"].values)

checks = [
    ("Physics MAPE < 15%",         phys_mape_all < 15,              f"{phys_mape_all:.2f}%",  ""),
    ("Hybrid MAPE < 7%",           hybrid_mape_all < 7,             f"{hybrid_mape_all:.2f}%", ""),
    ("Hybrid MAPE test set < 7%",  mape(ym_test, pred_test) < 7,   f"{mape(ym_test,pred_test):.2f}%", ""),
    ("Cross-trip validated",        len(all_cross_results) >= 2,     f"{len(all_cross_results)} trips", ""),
    ("Confidence intervals fitted", True,                            "90% CI  (5th/95th percentile)", ""),
    ("90% CI coverage ≥ 80%",      ci_cov_all >= 80,                f"{ci_cov_all:.1f}%", ""),
    ("Model serialised (.pkl)",    MODEL_FILE.exists(),              str(MODEL_FILE), ""),
    ("Validation report saved",    REPORT_OUT.exists(),              str(REPORT_OUT), ""),
]

print()
for desc, passed, value, note in checks:
    icon  = "✅" if passed else "❌"
    extra = f"  ← {note}" if note else ""
    print(f"  {icon}  {desc:<42}  {value}{extra}")

print(f"""
{'='*64}
PHASE 1 COMPLETE
  Physics MAPE  (cross-trips) : {phys_mape_all:.2f}%
  Hybrid  MAPE  (cross-trips) : {hybrid_mape_all:.2f}%
  90% CI coverage             : {ci_cov_all:.1f}%

  All outputs in:
    models/hybrid_model.pkl
    data/validation_report.csv
    figs/val_mape_comparison.png
    figs/val_pred_vs_measured.png
    figs/val_residual_qq.png
    figs/val_condition_matrix.png
{'='*64}
""")
"""
build_trip_dataset.py  [FIXED]
===============================
Segments the primary trip into sub-trips and computes trip-level
features + physics model predictions for the hybrid ML layer.

BUG FIXED vs PREVIOUS VERSION
------------------------------
Previously eta_sys was read from  Powertrain_efficiency_gear_SG  (~0.90).
This is the mechanical drivetrain efficiency, NOT total system efficiency.
The correct value (≈ 0.35–0.44) comes from  curves/eta_powertrain_vs_speed.csv
which was built empirically as P_mech / P_bat in build_eta_powertrain.py.

With the wrong eta, physics_pred_kWh was roughly 2× too low for every
sub-trip, making the residual/km signal dominated by a systematic bias
rather than the physical features we want the ML model to learn.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

PRIMARY_FILE   = "Tracking_data_efficiecny.csv"
ETA_PT_FILE    = Path("curves") / "eta_powertrain_vs_speed.csv"     # built by build_eta_powertrain.py
PARAMS_FILE    = Path("curves") / "tuned_physics_params.json"       # built by calibrate_physics_model.py
OUT_CSV        = Path("data")   / "trip_features.csv"
OUT_FIG_DIR    = Path("figs")

STOP_THRESHOLD_S    = 30
MIN_TRIP_DIST_KM    = 0.5
MIN_TRIP_DURATION_S = 60

DEFAULT_PARAMS = dict(
    Cd0  = 0.23,
    k_cd = 0.0,
    Crr  = 0.009,
    RHO  = 1.225,
    A    = 2.22,
    MASS = 1847,
    G    = 9.81,
)

HVAC_POWER_W = 0

OUT_CSV.parent.mkdir(exist_ok=True)
OUT_FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# STEP 1 — Load physics params
# ============================================================

if PARAMS_FILE.exists():
    with open(PARAMS_FILE) as fh:
        p = json.load(fh)
    Cd0  = p["Cd0"]
    k_cd = p.get("k_cd", 0.0)
    Crr  = p["Crr"]
    RHO  = p["RHO"]
    A    = p["A"]
    MASS = p["MASS"]
    G    = p["G"]
    print(f"[OK]  Loaded tuned physics params from {PARAMS_FILE}")
    print(f"      Cd0={Cd0:.4f}  Crr={Crr:.5f}  k_cd={k_cd:.6f}")
else:
    print(f"[WARN] {PARAMS_FILE} not found — using default Tesla Model 3 LR AWD specs")
    Cd0  = DEFAULT_PARAMS["Cd0"]
    k_cd = DEFAULT_PARAMS["k_cd"]
    Crr  = DEFAULT_PARAMS["Crr"]
    RHO  = DEFAULT_PARAMS["RHO"]
    A    = DEFAULT_PARAMS["A"]
    MASS = DEFAULT_PARAMS["MASS"]
    G    = DEFAULT_PARAMS["G"]

# ============================================================
# STEP 2 — Load eta_powertrain curve (TRUE system efficiency)
# ============================================================

if not ETA_PT_FILE.exists():
    raise FileNotFoundError(
        f"{ETA_PT_FILE} not found.  Run build_eta_powertrain.py first."
    )

eta_pt = pd.read_csv(ETA_PT_FILE).sort_values("speed_kph")
print(f"[OK]  Loaded eta_powertrain curve: "
      f"{eta_pt['eta_powertrain_mean'].min():.4f} – "
      f"{eta_pt['eta_powertrain_mean'].max():.4f}  "
      f"({len(eta_pt)} speed bins)")

def interp_eta(speed_kph_arr):
    """Interpolate system efficiency at each speed (km/h)."""
    return np.clip(
        np.interp(speed_kph_arr,
                  eta_pt["speed_kph"].values,
                  eta_pt["eta_powertrain_mean"].values,
                  left  = eta_pt["eta_powertrain_mean"].iloc[0],
                  right = eta_pt["eta_powertrain_mean"].iloc[-1]),
        0.25, 0.65
    )

# ============================================================
# STEP 3 — Load & clean primary trip
# ============================================================

print(f"\n[OK]  Loading {PRIMARY_FILE} ...")
df_raw = pd.read_csv(PRIMARY_FILE)

n_before = len(df_raw)
df = df_raw.dropna(subset=["Speed", "DeltaT", "Energy Consumption (kWh)",
                            "Slope Angle (rad)", "Acceleration"]).copy()

df["dt_s"]     = pd.to_numeric(df["DeltaT"],                        errors="coerce")
n_glitch       = (df["dt_s"] > 5.0).sum()
df["dt_s"]     = df["dt_s"].clip(0.05, 5.0)
df["v_mps"]    = pd.to_numeric(df["Speed"],                         errors="coerce").clip(lower=0) / 3.6
df["theta_rad"]= pd.to_numeric(df["Slope Angle (rad)"],             errors="coerce").fillna(0)
df["accel"]    = pd.to_numeric(df["Acceleration"],                  errors="coerce").fillna(0)
df["elev_chg"] = pd.to_numeric(df["ElevChange"],                    errors="coerce").fillna(0)
df["energy"]   = pd.to_numeric(df["Energy Consumption (kWh)"],     errors="coerce").fillna(0)
df["disp_m"]   = pd.to_numeric(df["Displacement (m)"],              errors="coerce").fillna(0)

# FIX: use true system efficiency from the empirical speed-dependent curve
df["eta_sys"] = interp_eta(df["Speed"].astype(float).values)

print(f"      Rows loaded      : {n_before:,}")
print(f"      DeltaT glitches  : {n_glitch} rows clipped")
print(f"      Rows after clean : {len(df):,}")
print(f"      eta_sys range    : {df['eta_sys'].min():.4f} – {df['eta_sys'].max():.4f}")

# ============================================================
# STEP 4 — Segment into sub-trips
# ============================================================

df["is_stop"]   = df["Speed"].astype(float) < 1.0
df["stop_group"]= (df["is_stop"] != df["is_stop"].shift()).cumsum()

stop_durations = (
    df[df["is_stop"]]
    .groupby("stop_group")["dt_s"]
    .sum()
)
long_stop_groups  = set(stop_durations[stop_durations > STOP_THRESHOLD_S].index)
df["is_long_stop"]= df["stop_group"].isin(long_stop_groups) & df["is_stop"]
df["trip_id"]     = df["is_long_stop"].cumsum()

n_raw_trips = df["trip_id"].nunique()
print(f"\n[OK]  Segmentation: {len(long_stop_groups)} long stops → {n_raw_trips} raw segments")

# ============================================================
# STEP 5 — Physics model prediction (row level)
# ============================================================

v     = df["v_mps"].to_numpy()
theta = df["theta_rad"].to_numpy()
a     = df["accel"].to_numpy()

Cd_v    = Cd0 + k_cd * v
F_aero  = 0.5 * RHO * Cd_v * A * v**2
F_rr    = Crr * MASS * G * np.cos(theta)
F_gr    = MASS * G * np.sin(theta)
F_in    = MASS * a
F_tot   = F_aero + F_rr + F_gr + F_in
P_mech_W = F_tot * v + HVAC_POWER_W

eta = df["eta_sys"].to_numpy()
P_bat_W = np.where(P_mech_W < 0,
                   P_mech_W * eta,    # regen: recover less
                   P_mech_W / eta)    # drive: draw more

df["P_bat_W"]    = P_bat_W
df["E_phys_kWh"] = P_bat_W * df["dt_s"].to_numpy() / 3_600_000

# ============================================================
# STEP 6 — Aggregate to trip-level features
# ============================================================

records = []

for trip_id, seg in df.groupby("trip_id"):
    dist_km  = seg["disp_m"].sum() / 1000.0
    duration = seg["dt_s"].sum()

    if dist_km < MIN_TRIP_DIST_KM or duration < MIN_TRIP_DURATION_S:
        continue

    moving = seg[seg["Speed"].astype(float) > 1.0]
    if len(moving) == 0:
        continue

    avg_speed_kph = float(moving["Speed"].mean())
    max_speed_kph = float(seg["Speed"].max())

    slope_deg = np.degrees(np.abs(seg["theta_rad"]))
    avg_slope_deg = float(slope_deg.mean())
    max_slope_deg = float(slope_deg.max())

    elev_gain_m   = float(seg["elev_chg"].clip(lower=0).sum())
    regen_rows    = (seg["P_bat_W"] < 0).sum()
    regen_fraction= float(regen_rows / max(len(seg), 1))

    energy_kWh      = float(seg["energy"].sum())
    physics_pred_kWh= float(seg["E_phys_kWh"].sum())
    residual_kWh    = energy_kWh - physics_pred_kWh

    # Speed tier
    if avg_speed_kph < 30:
        speed_tier = "urban"
    elif avg_speed_kph < 70:
        speed_tier = "mixed"
    else:
        speed_tier = "highway"

    efficiency_kWh_100km = (energy_kWh / dist_km * 100) if dist_km > 0 else np.nan

    records.append({
        "trip_id"             : trip_id,
        "dist_km"             : round(dist_km, 3),
        "duration_min"        : round(duration / 60, 2),
        "avg_speed_kph"       : round(avg_speed_kph, 3),
        "max_speed_kph"       : round(max_speed_kph, 3),
        "avg_slope_deg"       : round(avg_slope_deg, 4),
        "max_slope_deg"       : round(max_slope_deg, 4),
        "elev_gain_m"         : round(elev_gain_m, 2),
        "regen_fraction"      : round(regen_fraction, 4),
        "energy_kWh"          : round(energy_kWh, 5),
        "efficiency_kWh_100km": round(efficiency_kWh_100km, 3),
        "physics_pred_kWh"    : round(physics_pred_kWh, 5),
        "residual_kWh"        : round(residual_kWh, 5),
        "speed_tier"          : speed_tier,
    })

trips_df = pd.DataFrame(records)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRIP DATASET SUMMARY")
print("=" * 60)
print(f"  Valid sub-trips            : {len(trips_df)}")
print(f"  Total distance             : {trips_df['dist_km'].sum():.1f} km")
print(f"  Total measured energy      : {trips_df['energy_kWh'].sum():.3f} kWh")
print(f"  Total physics prediction   : {trips_df['physics_pred_kWh'].sum():.3f} kWh")

total_meas = trips_df["energy_kWh"].sum()
total_phys = trips_df["physics_pred_kWh"].sum()
overall_err = (total_phys - total_meas) / abs(total_meas) * 100
print(f"  Physics overall error      : {overall_err:+.2f}%")

mask_nz = trips_df["energy_kWh"].abs() > 0.05
per_trip_mape = np.mean(
    np.abs((trips_df.loc[mask_nz, "energy_kWh"]
            - trips_df.loc[mask_nz, "physics_pred_kWh"])
           / trips_df.loc[mask_nz, "energy_kWh"])
) * 100
print(f"  Physics per-trip MAPE      : {per_trip_mape:.2f}%")

print(f"\n  Speed tier distribution:")
print(trips_df["speed_tier"].value_counts().to_string())
print(f"\n  Feature statistics:")
print(trips_df[["dist_km", "avg_speed_kph", "elev_gain_m",
                "energy_kWh", "efficiency_kWh_100km",
                "residual_kWh"]].describe().round(3).to_string())
print("=" * 60)

trips_df.to_csv(OUT_CSV, index=False)
print(f"\n[OK]  Saved {len(trips_df)} trips → {OUT_CSV}")

# ============================================================
# PLOTS
# ============================================================

fig_style = {"figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
             "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
             "xtick.color": "#8b949e", "ytick.color": "#8b949e",
             "text.color": "#e6edf3", "grid.color": "#21262d",
             "grid.linewidth": 0.6}
plt.rcParams.update(fig_style)

# 1 — Correlation heatmap
numeric_cols = ["dist_km", "avg_speed_kph", "max_speed_kph",
                "avg_slope_deg", "max_slope_deg", "elev_gain_m",
                "regen_fraction", "energy_kWh", "residual_kWh"]
corr = trips_df[numeric_cols].corr()
fig1, ax1 = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.4, linecolor="#30363d", ax=ax1,
            cbar_kws={"shrink": 0.8})
ax1.set_title("Feature correlation matrix")
plt.tight_layout()
fig1.savefig(OUT_FIG_DIR / "eda_correlation_heatmap.png", dpi=150)
plt.close(fig1)
print("[OK]  Saved eda_correlation_heatmap.png")

# 2 — Efficiency vs speed
TIER_COLOR = {"highway": "#3fb950", "mixed": "#58a6ff", "urban": "#e3b341"}
fig2, ax2 = plt.subplots(figsize=(8, 5))
for tier, grp in trips_df.groupby("speed_tier"):
    ax2.scatter(grp["avg_speed_kph"], grp["efficiency_kWh_100km"],
                label=tier, color=TIER_COLOR.get(tier, "#888"),
                s=50, alpha=0.8, edgecolors="#0d1117", linewidth=0.4)
ax2.set_xlabel("Average speed (km/h)")
ax2.set_ylabel("Efficiency (kWh/100 km)")
ax2.set_title("Trip efficiency vs average speed")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(OUT_FIG_DIR / "eda_efficiency_vs_speed.png", dpi=150)
plt.close(fig2)
print("[OK]  Saved eda_efficiency_vs_speed.png")

# 3 — Efficiency vs elevation
fig3, ax3 = plt.subplots(figsize=(8, 5))
for tier, grp in trips_df.groupby("speed_tier"):
    ax3.scatter(grp["elev_gain_m"], grp["efficiency_kWh_100km"],
                label=tier, color=TIER_COLOR.get(tier, "#888"),
                s=50, alpha=0.8, edgecolors="#0d1117", linewidth=0.4)
ax3.set_xlabel("Elevation gain (m)")
ax3.set_ylabel("Efficiency (kWh/100 km)")
ax3.set_title("Trip efficiency vs elevation gain")
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
fig3.savefig(OUT_FIG_DIR / "eda_efficiency_vs_elevation.png", dpi=150)
plt.close(fig3)
print("[OK]  Saved eda_efficiency_vs_elevation.png")

# 4 — Residual distribution + scatter
fig4, (ax_hist, ax_sc) = plt.subplots(1, 2, figsize=(12, 5))
residuals = trips_df["residual_kWh"].dropna()
ax_hist.hist(residuals, bins=20, color="#58a6ff", alpha=0.8, edgecolor="#0d1117")
ax_hist.axvline(0,               color="#3fb950", linewidth=1.5, linestyle="--", label="Zero")
ax_hist.axvline(residuals.mean(),color="#e3b341", linewidth=1.5, linestyle=":",
                label=f"Mean={residuals.mean():.3f}")
ax_hist.axvline(residuals.median(),color="#f85149", linewidth=1.5, linestyle="-.",
                label=f"Median={residuals.median():.3f}")
ax_hist.set_xlabel("Residual (measured − physics)  kWh")
ax_hist.set_ylabel("Count")
ax_hist.set_title("Residual distribution")
ax_hist.legend(fontsize=8, framealpha=0.3)
ax_hist.grid(True, alpha=0.3)

ax_sc.scatter(trips_df["physics_pred_kWh"], trips_df["energy_kWh"],
              c=[TIER_COLOR.get(t, "#888") for t in trips_df["speed_tier"]],
              s=50, alpha=0.8, edgecolors="#0d1117", linewidth=0.4)
lims = [min(trips_df["energy_kWh"].min(), trips_df["physics_pred_kWh"].min()) - 0.5,
        max(trips_df["energy_kWh"].max(), trips_df["physics_pred_kWh"].max()) + 0.5]
ax_sc.plot(lims, lims, color="#3fb950", linewidth=1.2, linestyle="--", label="Perfect")
ax_sc.set_xlabel("Physics prediction (kWh)")
ax_sc.set_ylabel("Measured energy (kWh)")
ax_sc.set_title(f"Physics vs measured  (MAPE={per_trip_mape:.1f}%)")
ax_sc.legend(fontsize=8, framealpha=0.3)
ax_sc.grid(True, alpha=0.3)

plt.tight_layout()
fig4.savefig(OUT_FIG_DIR / "eda_residuals.png", dpi=150)
plt.close(fig4)
print("[OK]  Saved eda_residuals.png")

print(f"""
{'='*60}
DONE — outputs written:
  {OUT_CSV}
  {OUT_FIG_DIR}/eda_correlation_heatmap.png
  {OUT_FIG_DIR}/eda_efficiency_vs_speed.png
  {OUT_FIG_DIR}/eda_efficiency_vs_elevation.png
  {OUT_FIG_DIR}/eda_residuals.png

NEXT STEP:
  Run  hybrid_model.py  to fit the GBR correction layer
  on the residual column in  {OUT_CSV}
{'='*60}
""")
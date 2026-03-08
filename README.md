# Data-Driven Digital Twin — EV Energy Consumption Prediction
**Phase 1: Static Calibration & Model Freezing**
Final Year Project · Electrical / Mechanical / Computer Science

---

## Project Overview

This project builds a **hybrid physics + machine learning digital twin** that predicts electric vehicle energy consumption from trip parameters (speed, grade, distance, regeneration). The model is designed for **pre-trip journey planning** — a driver inputs their route conditions and receives an accurate energy estimate with a 90% confidence interval.

The vehicle modelled is a **Tesla Model 3 Long Range AWD** (mass 1847 kg, Cd 0.23, frontal area 2.22 m²).

**Key result:** Hybrid model MAPE of **4.77%** on completely unseen validation trips, against a physics-only baseline of ~127% (untuned) / ~12% (tuned).

---

## Repository Structure

```
├── curves/                          # Efficiency lookup tables (generated)
│   ├── eta_powertrain_vs_speed.csv
│   ├── eta_battery_vs_Tbat.csv
│   └── tuned_physics_params.json
│
├── data/                            # ML datasets (generated)
│   ├── trip_features.csv
│   └── validation_report.csv
│
├── models/                          # Trained model (generated)
│   └── hybrid_model.pkl
│
├── figs/                            # All output plots (generated)
│
├── 2_preprocessed/                  # Preprocessed intermediate data
│
├── Tracking_data_efficiecny.csv     # Primary trip — calibration (59,258 rows, 744 km)
├── trip_11apr2022.csv               # Validation trip 1 (1,818 rows, 28.5 km)
├── trip_12apr2022.csv               # Validation trip 2 (4,008 rows, 59.4 km)
│
├── build_eta_powertrain.py          # STEP 1a — build powertrain efficiency curve
├── build_eta_battery.py             # STEP 1b — build battery efficiency curve
├── calibrate_physics_model.py       # STEP 2  — tune Cd, Crr via L-BFGS-B
├── build_trip_dataset.py            # STEP 3  — segment trips, compute ML features
├── hybrid_model.py                  # STEP 4  — train GBR correction layer
├── ev_energy_model_validation_v3.py # STEP 5a — standalone physics model inspection
├── validate_hybrid_model.py         # STEP 5b — final cross-trip validation
└── README.md
```

---

## Run Order

> **Important:** scripts must be run in this exact order. Each step produces files that the next step depends on.

### Step 1 — Build efficiency curves
```bash
python build_eta_powertrain.py
python build_eta_battery.py
```
**Produces:** `curves/eta_powertrain_vs_speed.csv`, `curves/eta_battery_vs_Tbat.csv`

These are one-time lookup tables mapping vehicle speed → powertrain efficiency and battery temperature → battery efficiency. All downstream scripts interpolate from these curves.

---

### Step 2 — Calibrate physics model
```bash
python calibrate_physics_model.py
```
**Reads:** `Tracking_data_efficiecny.csv`, efficiency curves from Step 1
**Produces:** `curves/tuned_physics_params.json`

Uses L-BFGS-B optimisation to find the best values of the drag coefficient (Cd), rolling resistance (Crr), and speed-dependent drag correction (k_cd) by minimising RMSE between the parametric model and the data-driven mechanical power signal. Also validates the tuned model against all three trip files and saves a convergence plot.

---

### Step 3 — Build ML training dataset
```bash
python build_trip_dataset.py
```
**Reads:** `Tracking_data_efficiecny.csv`, `curves/tuned_physics_params.json`
**Produces:** `data/trip_features.csv`, four EDA plots in `figs/`

Segments the 59,258-row primary recording into **66 sub-trips** by detecting standstills longer than 30 seconds. Computes one feature row per sub-trip (average speed, distance, elevation gain, slope, regeneration fraction) and adds the physics model residual as the ML target column.

---

### Step 4 — Train hybrid model
```bash
python hybrid_model.py
```
**Reads:** `data/trip_features.csv`
**Produces:** `models/hybrid_model.pkl`, four model plots in `figs/`

Fits a Gradient Boosting Regressor to the **specific residual** (residual per km) using a 70/30 stratified train/test split and 5-fold GridSearchCV. Also fits 5th/95th percentile quantile models to produce 90% confidence intervals. Serialises the full model bundle (mean + quantile models, feature list, metrics) to a pickle file.

---

### Step 5a — Physics model inspection (optional)
```bash
python ev_energy_model_validation_v3.py
```
**Reads:** `Tracking_data_efficiecny.csv`, efficiency curves
**Produces:** diagnostic plots in `figs/`

Standalone script to inspect the physics model in isolation — cumulative energy comparison, instantaneous power plots, and a multi-axis diagnostic panel. Useful for verifying the physics layer before running the full hybrid pipeline.

---

### Step 5b — Final validation
```bash
python validate_hybrid_model.py
```
**Reads:** `models/hybrid_model.pkl`, `trip_11apr2022.csv`, `trip_12apr2022.csv`
**Produces:** `data/validation_report.csv`, four validation plots in `figs/`

Tests the frozen hybrid model on two completely unseen trips. Reports MAPE, RMSE, and R² broken down by speed tier (urban / mixed / highway), grade tier (flat / hilly), and trip length (short / medium / long). Prints a Phase 1 pass/fail checklist against the white paper success criteria.

---

## Results Summary

| Metric | Physics only | Hybrid model |
|---|---|---|
| Cross-trip MAPE | **9.12%** | **4.45%** |
| Test set MAPE | **7.37%** | **5.29%** |
| Test set R² | **0.9970** | **0.9989** |
| 90% CI coverage | — | **81.8%** *(cross-trips)* / **75.0%** *(internal test set)* |

> After Step 2 generates `tuned_physics_params.json`, the calibrated physics baseline reaches **9.12% cross-trip MAPE** on unseen trips.  
> The hybrid residual-correction model further reduces this to **4.45% cross-trip MAPE** and achieves **5.29% MAPE** on the internal held-out test set.  
> The final model also provides **90% confidence intervals**, with **81.8% coverage on unseen cross-trips**.

---

## Dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

Python 3.9 or higher recommended. No GPU required.

---

## Data Sources

- **Primary calibration trip:** `Tracking_data_efficiecny.csv` — real OBD-II recording, Cape Town, 5 April 2022, 744 km, 294.7 kWh
- **Validation trip 1:** `trip_11apr2022.csv` — 11 April 2022, 28.5 km, 11.5 kWh
- **Validation trip 2:** `trip_12apr2022.csv` — 12 April 2022, 59.4 km, 23.7 kWh

All trips recorded from a Tesla Model 3 LR AWD via GPS + OBD-II at 1-second intervals.

---

## Reference

Based on the project white paper:
> Banik, S. (2025). *Data-Driven Digital Twin for EV Energy Consumption Prediction: A Comprehensive Implementation Guide for Engineering Students.* November 2025.

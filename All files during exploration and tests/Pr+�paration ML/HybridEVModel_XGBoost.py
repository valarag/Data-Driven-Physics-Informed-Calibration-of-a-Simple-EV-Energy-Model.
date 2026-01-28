"""
HybridEVModel_XGBoost

Skeleton implementation of a Hybrid Physics + ML model for EV energy consumption.

Goal of this file (Preparation ML step):
- take a physics-based model as input,
- compute physics-only predictions for each trip,
- compute residuals (y_true - y_physics),
- prepare the feature matrix for a future XGBoost regressor.

IMPORTANT:
- No real XGBoost training is performed at this stage.
- The class is ready to receive residuals and features and will be extended later.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import xgboost as xgb  # noqa: F401
except ImportError:
    xgb = None


class HybridEVModel_XGBoost:
    """
    Hybrid Physics + ML model for EV energy consumption.
    """

    def __init__(self, physics_model, feature_cols: Optional[List[str]] = None, xgb_params: Optional[dict] = None):
        self.physics_model = physics_model
        self.feature_cols = feature_cols or [
            "avg_speed",
            "avg_temp",
            "payload",
            "grade",
            "distance_km",
        ]

        default_params = {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        if xgb_params is not None:
            default_params.update(xgb_params)
        self.xgb_params = default_params

        self.xgb_model = None
        self.is_trained = False

    def _physics_prediction_trip(self, row: pd.Series) -> float:
        if hasattr(self.physics_model, "predict_trip_energy"):
            return float(self.physics_model.predict_trip_energy(row))
        if hasattr(self.physics_model, "energy_per_100km"):
            e_per_100km = self.physics_model.energy_per_100km(
                avg_speed=row["avg_speed"],
                temp=row["avg_temp"],
                payload=row["payload"],
                grade=row["grade"],
            )
            return float(e_per_100km * row["distance_km"] / 100.0)
        raise AttributeError("physics_model must implement either 'predict_trip_energy' or 'energy_per_100km'.")

    def compute_physics_predictions(self, X_trips: pd.DataFrame) -> np.ndarray:
        return np.array([self._physics_prediction_trip(row) for _, row in X_trips.iterrows()], dtype=float)

    def compute_residuals(self, X_trips: pd.DataFrame, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_physics = self.compute_physics_predictions(X_trips)
        if len(y_true) != len(y_physics):
            raise ValueError("Length mismatch between y_true and physical predictions.")
        return y_physics, y_true - y_physics

    def prepare_training_data(self, X_trips: pd.DataFrame, y_true: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        missing_cols = [c for c in self.feature_cols if c not in X_trips.columns]
        if missing_cols:
            raise KeyError(f"Missing feature columns in input data: {missing_cols}")

        X_features = X_trips[self.feature_cols].copy()
        _, residuals = self.compute_residuals(X_trips, y_true)
        return X_features, residuals

    def predict(self, X_trips: pd.DataFrame) -> np.ndarray:
        y_physics = self.compute_physics_predictions(X_trips)
        if not self.is_trained or self.xgb_model is None:
            return y_physics
        return y_physics

    def fit(self, X_trips: pd.DataFrame, y_true: np.ndarray) -> None:
        pass

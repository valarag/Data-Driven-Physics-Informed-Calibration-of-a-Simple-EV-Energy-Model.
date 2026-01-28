import warnings
import xgboost as xgb
class HybridCalibrator:
   def fit(self, X, y):
       if len(X) < 10:
           warnings.warn("Not enough data for ML calibration")
           return
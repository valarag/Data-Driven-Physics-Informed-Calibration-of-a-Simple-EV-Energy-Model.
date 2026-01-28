import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings
class MeganePhysicsModel:
   def __init__(self, specs, curves):
       self.specs = specs
       self.curves = curves
       self._load_curves()
   def _load_curves(self):
       try:
           self.eta_pt = interp1d(
               self.curves["eta_powertrain"]["speed_kmph"],
               self.curves["eta_powertrain"]["eta_powertrain"],
               fill_value="extrapolate"
           )
       except Exception:
           warnings.warn("Powertrain efficiency curve missing")
           self.eta_pt = lambda x: 0.90
       try:
           self.eta_bat = interp1d(
               self.curves["eta_battery"]["T_battery_C"],
               self.curves["eta_battery"]["eta_battery"],
               fill_value="extrapolate"
           )
       except Exception:
           warnings.warn("Battery efficiency curve missing")
           self.eta_bat = lambda x: 0.95
       try:
           self.p_aux = interp1d(
               self.curves["p_aux"]["T_ambient_C"],
               self.curves["p_aux"]["P_aux_kw"],
               fill_value="extrapolate"
           )
       except Exception:
           warnings.warn("Auxiliary power curve missing")
           self.p_aux = lambda x: self.specs["baseline_aux_kw"]
   def compute_energy(self, trip_df):
       if trip_df.empty:
           warnings.warn("Trip data empty — returning zero energy")
           return 0.0
       # Placeholder simple integration
       return trip_df["E_total_kWh"].iloc[-1]
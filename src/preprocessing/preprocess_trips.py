import pandas as pd
import warnings
def preprocess_trips(path_raw, path_out):
   df = pd.read_csv(path_raw)
   if df.empty:
       warnings.warn("Raw trip file empty — writing empty processed file")
       df.to_csv(path_out, index=False)
       return
   # Placeholder
   df.to_csv(path_out, index=False)
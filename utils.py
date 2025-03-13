# utils.py
import pandas as pd
def validate_dataframe(df):
   """
   Ensures that the DataFrame is not empty and contains a 'Date' column.
   """
   if df.empty:
       raise ValueError("DataFrame is empty.")
   if 'Date' not in df.columns:
       raise ValueError("DataFrame does not have a 'Date' column.")
   return True
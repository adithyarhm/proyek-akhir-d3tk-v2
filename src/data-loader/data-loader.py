"""
DATA LOADER
=================
Menangani proses pemuatan file CSV node mentah, menggabungkannya menjadi satu DataFrame,
serta memuat data yang telah diproses sebelumnya.
"""

import pandas as pd
import os
from src.config import *

def load_and_merge_raw_nodes() -> pd.DataFrame:
    """
    Load file CSV tiap node dari direktori data raw,
    dan menggabungkannya menjadi satu DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame gabungan yang berisi seluruh data node.
    """

    df_combined = pd.DataFrame()
    for f in os.listdir(RAW_DATA_DIR):
        if f.endswith(".csv"):
            df_combined = pd.concat([df_combined, pd.read_csv(os.path.join(RAW_DATA_DIR, f))], ignore_index=True)
    return df_combined

def load_data() -> pd.DataFrame:
    """
    Memuat dataset akhir yang telah diproses (node_combined_final.csv).
    Dataset ini sudah siap digunakan untuk pelatihan setelah seluruh tahap
    preprocessing dan feature engineering diterapkan.
    
    Returns:
        pd.DataFrame: DataFrame akhir yang telah diproses.
    """
    df_final = pd.read_csv(DATA_PATH)
    if "datetime" in df_final.columns:
        df_final["datetime"] = pd.to_datetime(df_final["datetime"], errors="coerce")
    return df_final
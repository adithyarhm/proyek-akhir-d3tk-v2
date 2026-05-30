"""
PREPROCESSING & FEATURE ENGINEERING
=====================================
Konversi timestamp, sorting kronologis, rekayasa fitur waktu,
fitur turunan gas, dan penanganan missing values.
"""

import pandas as pd
import numpy as np
from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_PATH,
    TARGET_COL, BASE_FEATURES, TEMPORAL_FEATURES, DERIVED_FEATURES
)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline preprocessing utama:
    1. Parse & sort timestamp
    2. Rekayasa fitur temporal
    3. Rekayasa fitur turunan gas
    4. Tangani missing values
    5. Simpan ke processed dir
    """
    df = _parse_and_sort_datetime(df)
    df = _engineer_temporal_features(df)
    df = _engineer_derived_features(df)
    df = _handle_missing(df)
    _save_processed(df)
    return df


def _parse_and_sort_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse kolom timestamp ke datetime dan sort kronologis per node."""
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Buang baris dengan datetime tidak valid
    df = df.dropna(subset=["datetime"])

    # Sort per node secara kronologis — krusial untuk mencegah data leakage
    if "node_id" in df.columns:
        df = df.sort_values(["node_id", "datetime"]).reset_index(drop=True)
    else:
        df = df.sort_values("datetime").reset_index(drop=True)

    return df


def _engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ekstrak fitur waktu dari kolom datetime."""
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    # df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Senin
    # df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Encoding siklikal untuk jam (membantu model tangkap sifat periodik)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def _engineer_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rekayasa fitur turunan gas per node.
    Menggunakan diff() per node agar tidak terjadi bocoran antar node.
    """
    if "node_id" in df.columns:
        df["h2s_diff"] = df.groupby("node_id")["h2s_ugm"].diff().fillna(0)
        df["so2_diff"] = df.groupby("node_id")["so2_ugm"].diff().fillna(0)
    else:
        df["h2s_diff"] = df["h2s_ugm"].diff().fillna(0)
        df["so2_diff"] = df["so2_ugm"].diff().fillna(0)

    # Rasio SO2/H2S — indikator karakteristik emisi vulkanik
    eps = 1e-6  # hindari division by zero
    df["gas_ratio_so2_h2s"] = df["so2_ugm"] / (df["h2s_ugm"] + eps)

    # Lag features (t-1 dan t-2) untuk menangkap autokorelasi temporal
    for col in ["so2_ugm", "h2s_ugm"]:
        if "node_id" in df.columns:
            df[f"{col}_lag1"] = df.groupby("node_id")[col].shift(1).bfill()
            df[f"{col}_lag2"] = df.groupby("node_id")[col].shift(2).bfill()
        else:
            df[f"{col}_lag1"] = df[col].shift(1).bfill()
            df[f"{col}_lag2"] = df[col].shift(2).bfill()

    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Penanganan missing values:
    - Kolom numerik: forward-fill per node, lalu backward-fill sebagai fallback
    - Baris yang masih NaN di kolom target/fitur utama dihapus
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "node_id" in df.columns:
        df[numeric_cols] = (
            df.groupby("node_id")[numeric_cols]
            .transform(lambda x: x.ffill().bfill())
        )
    else:
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Drop baris yang kolom target-nya masih NaN
    df = df.dropna(subset=TARGET_COL).reset_index(drop=True)

    return df


def _save_processed(df: pd.DataFrame):
    """Simpan DataFrame hasil preprocessing ke direktori processed."""
    import os
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[Preprocess] Data tersimpan: {DATA_PATH} | Shape: {df.shape}")
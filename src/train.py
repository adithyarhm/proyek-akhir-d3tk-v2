"""
TRAINING PIPELINE
=================
Melatih 4 model regresi dengan dua skenario:
  - per_node : model terpisah per node (Skenario 1 & 2)
  - global   : satu model, semua node digabung (Skenario 3 & 4)

TimeSeriesSplit digunakan sebagai CV untuk mencegah data leakage.
Evaluasi metrik dilaporkan PER-NODE untuk perbandingan adil antar skenario.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.catboost import catboost_model
from src.models.lgbm import light_gbm_model
from src.models.xgboost import xgboost_model
from src.models.rfr import random_forest_regressor_model
from src.evaluation.metrics import metrics
from src.config import (
    TARGET_COL, SCENARIO_CONFIG, SCENARIO,
    BASE_FEATURES, TEMPORAL_FEATURES, DERIVED_FEATURES
)

N_SPLITS = 5  # jumlah fold TimeSeriesSplit


def get_feature_list(scenario_id: int) -> list:
    """Kembalikan daftar fitur sesuai skenario aktif."""
    cfg = SCENARIO_CONFIG[scenario_id]
    return cfg["features"]


def _build_models() -> dict:
    return {
        "CatBoost": catboost_model,
        "LightGBM": light_gbm_model,
        "XGBoost": xgboost_model,
        "RandomForest": random_forest_regressor_model,
    }


def train_per_node(df: pd.DataFrame, features: list, tuning_results: dict = None) -> dict:
    """
    Skenario Lokal: latih model terpisah per node.
    
    Returns:
        dict: {node_id: {model_name: {"model": obj, "metrics": {...}}}}
    """
    results = {}
    model_builders = _build_models()
    tss = TimeSeriesSplit(n_splits=N_SPLITS)

    for node_id, node_df in df.groupby("node_id"):
        node_df = node_df.sort_values("datetime").reset_index(drop=True)
        X = node_df[features].values
        y = node_df[TARGET_COL].values

        # Ambil split terakhir dari TimeSeriesSplit sebagai evaluasi final
        train_idx, test_idx = list(tss.split(X))[-1]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        results[node_id] = {}
        print(f"\n[Per-Node] Node: {node_id} | Train: {len(X_train)} | Test: {len(X_test)}")

        for model_name, builder in model_builders.items():
            if tuning_results and node_id in tuning_results and model_name in tuning_results[node_id]:
                model = tuning_results[node_id][model_name]
                model.fit(X_train, y_train)
            else:
                model = builder(X_train, y_train)

            y_pred = model.predict(X_test)
            m = metrics(y_test, y_pred)

            results[node_id][model_name] = {
                "model": model,
                "metrics": m,
                "y_test": y_test,
                "y_pred": y_pred,
            }
            print(f"  [{model_name}] RMSE={m['rmse']:.4f} | MAE={m['mae']:.4f} | R²={m['r2']:.4f}")

    return results


def train_global(df: pd.DataFrame, features: list, tuning_results: dict = None) -> dict:
    """
    Skenario Global: latih satu model pada data semua node.
    Koordinat spasial (lat, lon, elev) dimasukkan sebagai prediktor jika tersedia.
    Evaluasi tetap dilaporkan per-node untuk perbandingan adil.
    
    Returns:
        dict: {"global": {model_name: {"model": obj, "metrics": {node_id: {...}}}}}
    """
    # Tambahkan koordinat spasial jika ada di dataframe
    spatial_feats = [f for f in ["lat", "lon", "elev"] if f in df.columns]
    global_features = features + spatial_feats

    df_sorted = df.sort_values("datetime").reset_index(drop=True)
    X = df_sorted[global_features].values
    y = df_sorted[TARGET_COL].values

    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    train_idx, test_idx = list(tss.split(X))[-1]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model_builders = _build_models()
    results = {"global": {}}

    print(f"\n[Global] Train: {len(X_train)} | Test: {len(X_test)} | Features: {global_features}")

    for model_name, builder in model_builders.items():
        if tuning_results and "global" in tuning_results and model_name in tuning_results["global"]:
            model = tuning_results["global"][model_name]
            model.fit(X_train, y_train)
        else:
            model = builder(X_train, y_train)

        y_pred_all = model.predict(X_test)
        m_overall = metrics(y_test, y_pred_all)

        # Evaluasi per-node pada test set
        test_df = df_sorted.iloc[test_idx].copy()
        test_df["pred_so2"] = y_pred_all[:, 0]
        test_df["pred_h2s"] = y_pred_all[:, 1]

        per_node_metrics = {}
        for node_id, node_test in test_df.groupby("node_id"):
            yt = node_test[TARGET_COL].values
            yp = node_test[["pred_so2", "pred_h2s"]].values
            per_node_metrics[node_id] = metrics(yt, yp)

        results["global"][model_name] = {
            "model": model,
            "metrics_overall": m_overall,
            "metrics_per_node": per_node_metrics,
            "y_test": y_test,
            "y_pred": y_pred_all,
            "features_used": global_features,
        }
        print(f"  [{model_name}] RMSE={m_overall['rmse']:.4f} | MAE={m_overall['mae']:.4f} | R²={m_overall['r2']:.4f}")

    return results


def run_training(df: pd.DataFrame, tuning_results: dict = None) -> dict:
    """
    Entry point pelatihan. Otomatis memilih skenario dari config.py.
    """
    cfg = SCENARIO_CONFIG[SCENARIO]
    mode = cfg["mode"]
    features = cfg["features"]

    print(f"\n{'='*55}")
    print(f"  Skenario {SCENARIO}: {cfg['name']}")
    print(f"  Mode: {mode.upper()} | Fitur: {features}")
    print(f"{'='*55}")

    if mode == "per_node":
        return train_per_node(df, features, tuning_results)
    elif mode == "global":
        return train_global(df, features, tuning_results)
    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")
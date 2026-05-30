"""
EVALUATION METRICS
==================
Menghitung metrik evaluasi untuk model regresi multi-output.
Metrik dihitung secara keseluruhan dan per-target (H2S, SO2).
"""

import numpy as np
import pandas as pd
from src.config import TARGET_COL
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    mask = y_true != 0  # hindari division by zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
    }

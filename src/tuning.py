"""
HYPERPARAMETER TUNING (RandomizedSearchCV)
==========================================
Skrip untuk hyperparameter tuning menggunakan RandomizedSearchCV
pada semua model. Mendukung mode per_node dan global sesuai skenario aktif.
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Definisi search space untuk setiap model
PARAM_DISTRIBUTIONS = {
    "rfr": {
        "estimator__n_estimators": [50, 100, 200, 300, 500],
        "estimator__max_depth": [None, 5, 10, 15, 20, 30],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__min_samples_leaf": [1, 2, 4],
        "estimator__max_features": ["sqrt", "log2", None],
        "estimator__bootstrap": [True, False],
    },
    "xgbr": {
        "estimator__n_estimators": [50, 100, 200, 300, 500],
        "estimator__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "estimator__max_depth": [3, 5, 6, 8, 10],
        "estimator__min_child_weight": [1, 3, 5, 7],
        "estimator__gamma": [0, 0.1, 0.2, 0.3],
        "estimator__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "estimator__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "lgbm": {
        "estimator__n_estimators": [50, 100, 200, 300, 500],
        "estimator__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "estimator__num_leaves": [15, 31, 50, 70, 100],
        "estimator__max_depth": [-1, 5, 10, 15, 20],
        "estimator__min_child_samples": [5, 10, 20, 30],
        "estimator__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "estimator__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "cat": {
        "estimator__iterations": [50, 100, 200, 300, 500],
        "estimator__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "estimator__depth": [4, 6, 8, 10],
        "estimator__l2_leaf_reg": [1, 3, 5, 7, 10],
        "estimator__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
}

# Mapping nama model -> base estimator
MODEL_ESTIMATORS = {
    "rfr": lambda: MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1, verbose=0)),
    "xgbr": lambda: MultiOutputRegressor(XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)),
    "lgbm": lambda: MultiOutputRegressor(LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)),
    "cat": lambda: MultiOutputRegressor(CatBoostRegressor(random_state=42, silent=True)),
}

MODEL_DISPLAY = {
    "rfr": "Random Forest Regressor",
    "xgbr": "XGBoost",
    "lgbm": "LightGBM",
    "cat": "CatBoost",
}

def run_tuning (X, y, mode):
    """
    Menjalankan hyperparameter tuning untuk semua model
    Mode 'per_node' untuk tuning per node
    Mode 'global' untuk tuning global
    """
    
    # Konfigurasi RandomizedSearchCV
    N_ITER = 50
    CV = TimeSeriesSplit(n_splits=5)
    SCORING = "neg_mean_absolute_error"  # metrik evaluasi
    
    best_models = {}
    
    if mode == "per_node":
        # Tuning per node
        for node in X.columns:
            print(f"Tuning untuk node {node}...")
            X_node = X[node].values.reshape(-1, 1)
            y_node = y[node].values.reshape(-1, 1)
            
            for model_name in MODEL_ESTIMATORS:
                print(f"  Training model {MODEL_DISPLAY[model_name]}...")
                
                # Setup RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    estimator=MODEL_ESTIMATORS[model_name](),
                    param_distributions=PARAM_DISTRIBUTIONS[model_name],
                    n_iter=N_ITER,
                    cv=CV,
                    scoring=SCORING,
                    n_jobs=-1,
                    verbose=0,
                )
                
                # Run tuning
                random_search.fit(X_node, y_node)
                
                # Store best model
                best_models[node, model_name] = random_search.best_estimator_
                
                print(f"    Best MAE: {-random_search.best_score_:.4f}")
                print(f"    Best params: {random_search.best_params_}\n")
    
    elif mode == "global":
        # Tuning global (menggabungkan semua node)
        print("Running global hyperparameter tuning...")
        
        # Gabungkan semua node menjadi satu dataset
        X_global = X.values
        y_global = y.values
        
        for model_name in MODEL_ESTIMATORS:
            print(f"  Training model {MODEL_DISPLAY[model_name]}...")
            
            # Setup RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=MODEL_ESTIMATORS[model_name](),
                param_distributions=PARAM_DISTRIBUTIONS[model_name],
                n_iter=N_ITER,
                cv=CV,
                scoring=SCORING,
                n_jobs=-1,
                verbose=0,
            )
            
            # Run tuning
            random_search.fit(X_global, y_global)
            
            # Store best model (global)
            best_models["global", model_name] = random_search.best_estimator_
            
            print(f"    Best MAE: {-random_search.best_score_:.4f}")
            print(f"    Best params: {random_search.best_params_}\n")
    
    else:
        raise ValueError("Mode harus 'per_node' atau 'global'")
    
    return best_models
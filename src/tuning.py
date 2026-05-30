"""
HYPERPARAMETER TUNING — OPTUNA (TPE Bayesian Optimization)
===========================================================
Strategi: Tree-structured Parzen Estimator (TPE) dengan MedianPruner.
Setiap trial belajar dari trial sebelumnya → jauh lebih efisien dari
RandomizedSearchCV yang sampling secara acak murni.

Mendukung dua mode:
  - per_node : study terpisah per node (Skenario 1 & 2)
  - global   : satu study untuk data gabungan (Skenario 3 & 4)

Output: dict {model_name: fitted_best_model} siap dioper ke run_training()
"""

import os
import json
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.config import TARGET_COL

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Konfigurasi Global ──────────────────────────────────────────────────────
N_TRIALS    = 50    # Naikkan ke 100+ untuk hasil lebih optimal
N_SPLITS_CV = 5     # Fold TimeSeriesSplit untuk CV dalam setiap trial
N_STARTUP   = 10    # Trial random sebelum TPE aktif
RESULTS_DIR = "outputs/tuning"


# ── Search Space per Algoritma ──────────────────────────────────────────────

def _suggest_xgboost(trial) -> dict:
    """
    n_estimators     : Jumlah boosting rounds.
    learning_rate    : Shrinkage per tree. log=True karena efeknya eksponensial.
    max_depth        : Kedalaman tree. 3-8 optimal untuk data tabular/sensor.
    subsample        : Fraksi data per tree → regularisasi stokastik.
    colsample_bytree : Fraksi fitur per tree → kurangi korelasi antar tree.
    min_child_weight : Min sum hessian di leaf → cegah split pada sampel langka.
    gamma            : Min gain untuk split → pruning threshold.
    reg_alpha        : L1 regularisasi → dorong weight leaf ke nol (sparsity).
    reg_lambda       : L2 regularisasi → haluskan weight leaf.
    """
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42, "n_jobs": -1, "verbosity": 0,
    }


def _suggest_lightgbm(trial) -> dict:
    """
    num_leaves : PARAMETER KRITIS LightGBM. Mengontrol kompleksitas pohon
                 karena LightGBM pakai leaf-wise growth (bukan level-wise).
                 Constraint ketat: num_leaves < 2^max_depth untuk cegah overfit.
    min_child_samples : Min data di satu leaf. Penting untuk data sensor IoT
                        yang distribusinya tidak merata (gas spike jarang terjadi).
    """
    max_depth  = trial.suggest_int("max_depth", 3, 12)
    max_leaves = min(2 ** max_depth - 1, 255)
    num_leaves = trial.suggest_int("num_leaves", 15, max_leaves)

    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves":        num_leaves,
        "max_depth":         max_depth,
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42, "n_jobs": -1, "verbosity": -1,
    }


def _suggest_catboost(trial) -> dict:
    """
    depth : CatBoost pakai symmetric trees → semua split di level yang sama
            menggunakan fitur yang sama. Jauh lebih cepat dari XGB/LGB,
            tapi depth 6-8 sudah sangat ekspresif.
    l2_leaf_reg     : L2 pada leaf values.
    random_strength : Noise pada split scoring di awal training → regularisasi implisit.
    bagging_temperature : Intensitas Bayesian bagging.
                          0=deterministik, 1=stochastic penuh.
    border_count    : Candidate splits untuk fitur numerik.
                      Lebih tinggi = presisi lebih baik, lebih lambat.
    """
    return {
        "iterations":          trial.suggest_int("iterations", 100, 600, step=50),
        "learning_rate":       trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth":               trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        "random_strength":     trial.suggest_float("random_strength", 0.0, 3.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count":        trial.suggest_int("border_count", 32, 255),
        "random_state": 42, "silent": True, "allow_writing_files": False,
    }


def _suggest_random_forest(trial) -> dict:
    """
    max_depth : RF lebih stabil terhadap depth besar dibanding boosting,
                namun tetap perlu dibatasi untuk data sensor yang noisy.
    min_samples_leaf : Kontrol ukuran leaf secara langsung → lebih intuitif
                       dari min_samples_split untuk regression tasks.
    max_features : 'sqrt' = standar RF. 'log2' = lebih agresif feature sampling.
                   None = semua fitur (risk overfit pada dataset kecil).
    """
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth":         trial.suggest_int("max_depth", 3, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": 42, "n_jobs": -1,
    }


# ── Registry Model ──────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "XGBoost":      (XGBRegressor,          _suggest_xgboost),
    "LightGBM":     (LGBMRegressor,         _suggest_lightgbm),
    "CatBoost":     (CatBoostRegressor,     _suggest_catboost),
    "RandomForest": (RandomForestRegressor, _suggest_random_forest),
}


# ── Core Objective Function ─────────────────────────────────────────────────

def _build_objective(X: np.ndarray, y: np.ndarray, model_name: str):
    """
    Factory fungsi objective untuk satu model.
    Setiap trial: suggest params → build model → TimeSeriesSplit CV → return MAE.
    Optuna meminimalkan nilai return menggunakan TPE.
    """
    ModelClass, suggest_fn = MODEL_REGISTRY[model_name]
    cv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

    def objective(trial):
        params = suggest_fn(trial)
        model  = MultiOutputRegressor(ModelClass(**params))
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            error_score="raise",
        )
        return -scores.mean()  # Optuna minimize → kembalikan positif MAE

    return objective


# ── Fungsi Tuning Utama ─────────────────────────────────────────────────────

def tune_single(X: np.ndarray, y: np.ndarray,
                model_name: str, label: str = "global",
                n_trials: int = N_TRIALS) -> dict:
    """
    Jalankan Optuna study untuk satu model.

    Returns:
        dict: {
            "model"      : fitted MultiOutputRegressor dengan best params,
            "best_params": dict hyperparameter terbaik,
            "best_mae"   : float nilai MAE terbaik,
            "study"      : objek optuna.Study untuk visualisasi/analisis lanjut,
        }
    """
    ModelClass, _ = MODEL_REGISTRY[model_name]

    study = optuna.create_study(
        study_name=f"{label}_{model_name}",
        direction="minimize",
        sampler=TPESampler(
            seed=42,
            n_startup_trials=N_STARTUP,  # trial pertama random sebelum TPE aktif
            multivariate=True,           # pertimbangkan korelasi antar parameter
        ),
        pruner=MedianPruner(
            n_startup_trials=N_STARTUP,
            n_warmup_steps=5,            # tunggu 5 step sebelum prune
        ),
    )

    study.optimize(
        _build_objective(X, y, model_name),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )

    best_params = study.best_params
    best_mae    = study.best_value

    print(f"    Best MAE : {best_mae:.4f}")
    print(f"    Best Params : {best_params}")

    # Rebuild & fit pada seluruh data dengan best_params
    best_model = MultiOutputRegressor(ModelClass(**best_params))
    best_model.fit(X, y)

    return {
        "model":       best_model,
        "best_params": best_params,
        "best_mae":    best_mae,
        "study":       study,
    }


def run_tuning(df, features: list, mode: str,
               n_trials: int = N_TRIALS) -> dict:
    """
    Entry point tuning. Dipanggil dari main.py sebelum run_training().

    Args:
        df       : DataFrame hasil preprocessing (sudah ada kolom node_id, datetime).
        features : List nama fitur sesuai skenario aktif.
        mode     : "per_node" atau "global".
        n_trials : Jumlah trial Optuna per model.

    Returns:
        dict kompatibel dengan parameter tuning_results di train.py.
        per_node → {node_id: {model_name: fitted_model}}
        global   → {"global": {model_name: fitted_model}}
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tuning_results = {}
    spatial_feats  = [f for f in ["lat", "lon", "elev"] if f in df.columns]

    print(f"\n{'='*60}")
    print(f"  OPTUNA HYPERPARAMETER TUNING")
    print(f"  Mode: {mode.upper()} | Trials per model: {n_trials}")
    print(f"  Sampler: TPE multivariate | CV: TimeSeriesSplit({N_SPLITS_CV})")
    print(f"{'='*60}")

    if mode == "per_node":
        for node_id, node_df in df.groupby("node_id"):
            node_df = node_df.sort_values("datetime").reset_index(drop=True)
            X = node_df[features].values
            y = node_df[TARGET_COL].values
            label = f"node{node_id}"

            print(f"\n[Tuning] Node: {node_id} | Samples: {len(X)}")
            tuning_results[node_id] = {}

            for model_name in MODEL_REGISTRY:
                print(f"  → {model_name} ({n_trials} trials)...")
                result = tune_single(X, y, model_name, label, n_trials)
                tuning_results[node_id][model_name] = result["model"]
                _save_best_params(result["best_params"], result["best_mae"],
                                  model_name, label)

    elif mode == "global":
        global_features = features + spatial_feats
        df_sorted = df.sort_values("datetime").reset_index(drop=True)
        avail = [f for f in global_features if f in df_sorted.columns]
        X = df_sorted[avail].values
        y = df_sorted[TARGET_COL].values

        print(f"\n[Tuning] Global | Samples: {len(X)} | Features: {avail}")
        tuning_results["global"] = {}

        for model_name in MODEL_REGISTRY:
            print(f"  → {model_name} ({n_trials} trials)...")
            result = tune_single(X, y, model_name, "global", n_trials)
            tuning_results["global"][model_name] = result["model"]
            _save_best_params(result["best_params"], result["best_mae"],
                              model_name, "global")

    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")

    print(f"\n[Tuning] Selesai. Params tersimpan di: {RESULTS_DIR}/")
    return tuning_results


# ── Helper: Simpan Best Params ──────────────────────────────────────────────

def _save_best_params(params: dict, best_mae: float,
                      model_name: str, label: str):
    """Simpan hyperparameter terbaik ke JSON untuk reproducibility."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    record = {"model": model_name, "label": label,
              "best_mae": best_mae, "params": params}
    path = os.path.join(RESULTS_DIR, f"best_params_{label}_{model_name}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)


# ── Utilitas Visualisasi Diagnostik Optuna ──────────────────────────────────

def plot_optuna_results(study: optuna.Study, model_name: str,
                        label: str = "global",
                        output_dir: str = "outputs/tuning/plots"):
    """
    Hasilkan 3 plot diagnostik interaktif:
      1. optimization_history  → konvergensi MAE per trial
      2. param_importances     → seberapa penting setiap hyperparameter
      3. parallel_coordinate   → visualisasi multi-dimensi search space
    """
    try:
        import optuna.visualization as vis
        import plotly.io as pio
        os.makedirs(output_dir, exist_ok=True)

        plots = {
            "history":    vis.plot_optimization_history(study),
            "importance": vis.plot_param_importances(study),
            "parallel":   vis.plot_parallel_coordinate(study),
        }
        for name, fig in plots.items():
            fig.update_layout(title=f"{model_name} — {name} ({label})")
            path = f"{output_dir}/{label}_{model_name}_{name}.html"
            pio.write_html(fig, path)
            print(f"  [Plot] {path}")

    except ImportError:
        print("  [Info] Install plotly: pip install plotly fish")
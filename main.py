"""
MAIN PIPELINE
=============
Orkestrasi fase penelitian:
  Fase 1 → Preprocessing
  Fase 2 → Training (per-node / global sesuai SCENARIO di config.py)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import SCENARIO, SCENARIO_CONFIG, TARGET_COL
from src.data.data_loader import load_and_merge_raw_nodes, load_data
from src.tuning import run_tuning
from src.features.preprocess import preprocess
from src.train import run_training, get_feature_list
from src.savemodels import save_all_models
from src.tuning import run_tuning, plot_optuna_results


def print_metrics_table(results: dict, mode: str):
    """Cetak tabel metrik komprehensif ke console."""
    rows = []
    if mode == "per_node":
        for node_id, node_res in results.items():
            for mname, mdata in node_res.items():
                m = mdata["metrics"]
                rows.append({
                    "Node": node_id, "Model": mname,
                    "RMSE": round(m["rmse"], 4), "MAE": round(m["mae"], 4),
                    "R²": round(m["r2"], 4),  "MAPE": round(m["mape"], 2)
                })
    else:
        for mname, mdata in results["global"].items():
            m = mdata["metrics_overall"]
            rows.append({
                "Node": "GLOBAL", "Model": mname,
                "RMSE": round(m["rmse"], 4), "MAE": round(m["mae"], 4),
                "R²": round(m["r2"], 4), "MAPE": round(m["mape"], 2)
            })
    df_table = pd.DataFrame(rows)
    print("\n" + "="*65)
    print("  RINGKASAN METRIK EVALUASI")
    print("="*65)
    print(df_table.to_string(index=False))
    df_table.to_csv(f"outputs/metrics_scenario_{SCENARIO}.csv", index=False)
    print(f"\n[Main] Metrik tersimpan: outputs/metrics_scenario_{SCENARIO}.csv")


def plot_metrics_barchart(results: dict, mode: str, output_dir: str = "outputs/plots"):
    """Bar chart perbandingan RMSE dan MAE antar model."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    if mode == "per_node":
        for node_id, node_res in results.items():
            for mname, mdata in node_res.items():
                m = mdata["metrics"]
                rows.append({"Model": mname, "RMSE": m["rmse"], "MAE": m["mae"], "Node": str(node_id)})
    else:
        for mname, mdata in results["global"].items():
            m = mdata["metrics_overall"]
            rows.append({"Model": mname, "RMSE": m["rmse"], "MAE": m["mae"], "Node": "Global"})

    df_bar = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        if mode == "per_node":
            df_pivot = df_bar.pivot(index="Model", columns="Node", values=metric)
            df_pivot.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white")
        else:
            ax.bar(df_bar["Model"], df_bar[metric], color=plt.cm.tab10.colors[:len(df_bar)])

        ax.set_title(f"Perbandingan {metric} — Skenario {SCENARIO}")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    path = f"{output_dir}/metrics_barchart_scenario_{SCENARIO}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Main] Bar chart metrik tersimpan: {path}")


def main():
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    cfg = SCENARIO_CONFIG[SCENARIO]
    mode = cfg["mode"]
    features = get_feature_list(SCENARIO)

    print(f"\n{'#'*60}")
    print(f"  KAWAH PUTIH GAS PREDICTION SYSTEM")
    print(f"  Skenario {SCENARIO}: {cfg['name']}")
    print(f"{'#'*60}")

    # ── FASE 1: Preprocessing ─────────────────────────────────────
    print("\n[FASE 1] Preprocessing Data...")
    try:
        df = load_data()
        print(f"  Dataset sudah diproses tersedia: {df.shape}")
    except FileNotFoundError:
        print("  File processed tidak ditemukan. Jalankan preprocessing...")
        df_raw = load_and_merge_raw_nodes()
        df = preprocess(df_raw)

    print(f"  Kolom tersedia: {list(df.columns)}")
    print(f"  Fitur yang digunakan: {features}")


    # ── FASE 2a: Tuning (opsional, aktifkan dengan flag) ──────────────
    USE_TUNING = True
    
    N_TUNING_TRIALS = 10

    tuning_results = None
    if USE_TUNING:
        print("\n[FASE 2a] Optuna Hyperparameter Tuning...")
        tuning_results = run_tuning(
            df=df,
            features=features,
            mode=mode,
            n_trials=N_TUNING_TRIALS,
        )

    # ── FASE 2b: Training dengan hasil tuning ────────────────────────
    print("\n[FASE 2b] Training Model...")
    results = run_training(df, tuning_results=tuning_results)
    print_metrics_table(results, mode)
    plot_metrics_barchart(results, mode)

    # ── Simpan semua model pemenang ───────────────────────────────
    print("\n[Main] Menyimpan model...")
    save_all_models(results, mode)

    # ── Tentukan model terbaik ────────────────────────────────────
    if mode == "global":
        best_name = min(
            results["global"],
            key=lambda m: results["global"][m]["metrics_overall"]["rmse"],
        )
        best_rmse = results["global"][best_name]["metrics_overall"]["rmse"]
    else:  # per_node
        # Rata-rata RMSE tiap model di seluruh node
        model_names = set()
        for node_res in results.values():
            model_names.update(node_res.keys())
        avg_rmse = {}
        for m in model_names:
            rmses = [results[n][m]["metrics"]["rmse"] for n in results if m in results[n]]
            avg_rmse[m] = sum(rmses) / len(rmses)
        best_name = min(avg_rmse, key=avg_rmse.get)
        best_rmse = avg_rmse[best_name]

    print(f"\n{'#'*60}")
    print(f"  PIPELINE SELESAI — Skenario {SCENARIO}: {cfg['name']}")
    print(f"  Model Terbaik: {best_name} | RMSE: {best_rmse:.4f}")
    print(f"  Output tersimpan di: outputs/")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
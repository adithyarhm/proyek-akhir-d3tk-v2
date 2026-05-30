"""
INFERENCE — Prediksi 1 Jam ke Depan
====================================
Menggunakan model terbaik untuk memprediksi konsentrasi SO2 dan H2S
selama 1 jam ke depan dengan pendekatan rolling forecast:
  - Setiap prediksi menggunakan output sebelumnya sebagai lag features
  - Fitur cuaca diasumsikan konstan (nilai terakhir yang diketahui)
  - Fitur temporal dihitung berdasarkan timestamp yang di-generate

Output:
  - CSV prediksi: outputs/inference_1h_scenario_{SCENARIO}.csv
  - Plot prediksi: outputs/plots/inference_1h_scenario_{SCENARIO}.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    SCENARIO, SCENARIO_CONFIG, TARGET_COL,
    TEMPORAL_FEATURES, DERIVED_FEATURES,
)

# Interval antar data dalam dataset (4 detik)
DATA_INTERVAL_SECONDS = 4

# Durasi prediksi ke depan: 1 jam = 3600 detik
FORECAST_HORIZON_SECONDS = 3600

# Jumlah langkah prediksi
N_FORECAST_STEPS = FORECAST_HORIZON_SECONDS // DATA_INTERVAL_SECONDS  # = 900


def _get_best_model(results: dict, mode: str):
    """Ambil model terbaik dan namanya dari results training."""
    if mode == "global":
        best_name = min(
            results["global"],
            key=lambda m: results["global"][m]["metrics_overall"]["rmse"],
        )
        best_model = results["global"][best_name]["model"]
        return best_name, best_model
    else:
        # Per-node: ambil model dengan rata-rata RMSE terendah
        model_names = set()
        for node_res in results.values():
            model_names.update(node_res.keys())
        avg_rmse = {}
        for m in model_names:
            rmses = [
                results[n][m]["metrics"]["rmse"]
                for n in results if m in results[n]
            ]
            avg_rmse[m] = sum(rmses) / len(rmses)
        best_name = min(avg_rmse, key=avg_rmse.get)
        # Ambil model dari node pertama (semua node dilatih dengan hyperparameter sama)
        first_node = list(results.keys())[0]
        best_model = results[first_node][best_name]["model"]
        return best_name, best_model


def _build_future_features(
    df_node: pd.DataFrame,
    features_used: list,
    n_steps: int,
    model,
    mode: str,
) -> pd.DataFrame:
    """
    Bangun DataFrame fitur untuk n_steps ke depan dengan rolling forecast.
    
    Langkah:
    1. Ambil state terakhir (baris terakhir) dari data aktual
    2. Generate timestamp baru tiap 4 detik
    3. Untuk setiap step:
       - Hitung fitur temporal dari timestamp
       - Gunakan prediksi sebelumnya sebagai lag features
       - Pertahankan fitur cuaca konstan (last known)
       - Prediksi SO2 & H2S
    """
    last_row = df_node.iloc[-1].copy()
    last_datetime = pd.to_datetime(last_row["datetime"])

    # State awal dari data aktual
    so2_prev1 = last_row.get("so2_ugm", 0.0)
    so2_prev2 = last_row.get("so2_ugm_lag1", so2_prev1)
    h2s_prev1 = last_row.get("h2s_ugm", 0.0)
    h2s_prev2 = last_row.get("h2s_ugm_lag1", h2s_prev1)

    # Fitur cuaca konstan (last known values)
    temp_c = last_row.get("temp_c", 0.0)
    hum_pct = last_row.get("hum_pct", 0.0)
    wind_kph = last_row.get("wind_kph", 0.0)

    # Koordinat spasial (konstan)
    lat = last_row.get("lat", 0.0)
    lon = last_row.get("lon", 0.0)
    elev = last_row.get("elev", 0.0)

    records = []

    for step in range(1, n_steps + 1):
        future_dt = last_datetime + pd.Timedelta(seconds=DATA_INTERVAL_SECONDS * step)

        # Fitur temporal
        hour = future_dt.hour
        minute = future_dt.minute
        minute_of_day = hour * 60 + minute

        # Fitur turunan dari prediksi sebelumnya
        so2_diff = so2_prev1 - so2_prev2 if step > 1 else 0.0
        h2s_diff = h2s_prev1 - h2s_prev2 if step > 1 else 0.0
        eps = 1e-6
        gas_ratio = so2_prev1 / (h2s_prev1 + eps)

        # Susun fitur sesuai urutan features_used
        feature_map = {
            "hum_pct": hum_pct,
            "temp_c": temp_c,
            "wind_kph": wind_kph,
            "hour": hour,
            "minute": minute,
            "minute_of_day": minute_of_day,
            "h2s_diff": h2s_diff,
            "so2_diff": so2_diff,
            "gas_ratio_so2_h2s": gas_ratio,
            "so2_ugm_lag1": so2_prev1,
            "so2_ugm_lag2": so2_prev2,
            "h2s_ugm_lag1": h2s_prev1,
            "h2s_ugm_lag2": h2s_prev2,
            "lat": lat,
            "lon": lon,
            "elev": elev,
        }

        X_row = np.array([[feature_map.get(f, 0.0) for f in features_used]])
        y_pred = model.predict(X_row)  # shape: (1, 2) -> [so2, h2s]

        so2_pred = float(y_pred[0, 0])
        h2s_pred = float(y_pred[0, 1])

        # Clamp ke non-negatif
        so2_pred = max(0.0, so2_pred)
        h2s_pred = max(0.0, h2s_pred)

        records.append({
            "datetime": future_dt,
            "step": step,
            "so2_pred": round(so2_pred, 4),
            "h2s_pred": round(h2s_pred, 4),
            "temp_c": round(temp_c, 2),
            "hum_pct": round(hum_pct, 2),
            "wind_kph": round(wind_kph, 2),
        })

        # Update lag state untuk langkah berikutnya
        so2_prev2 = so2_prev1
        so2_prev1 = so2_pred
        h2s_prev2 = h2s_prev1
        h2s_prev1 = h2s_pred

    return pd.DataFrame(records)


def _plot_inference(
    df_history: pd.DataFrame,
    df_forecast: pd.DataFrame,
    model_name: str,
    node_label: str,
    output_dir: str = "outputs/plots",
) -> str:
    """Buat plot gabungan historis + prediksi 1 jam ke depan."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, facecolor="#f7f6f2")

    # Ambil 15 menit terakhir historis sebagai konteks
    hist_tail = df_history.tail(225)  # ~15 menit @ 4 detik/sample
    hist_dt = pd.to_datetime(hist_tail["datetime"])

    forecast_dt = df_forecast["datetime"]

    targets_info = [
        ("so2_ugm", "so2_pred", "SO2 (ug/m3)", "#e67e22", "#e74c3c"),
        ("h2s_ugm", "h2s_pred", "H2S (ug/m3)", "#3498db", "#2980b9"),
    ]

    for ax, (hist_col, pred_col, label, color_hist, color_pred) in zip(axes, targets_info):
        ax.set_facecolor("#faf9f7")

        # Historis
        ax.plot(
            hist_dt, hist_tail[hist_col],
            color=color_hist, linewidth=1.5, alpha=0.8,
            label=f"{label} (Aktual)"
        )

        # Garis vertikal pembatas
        boundary = hist_dt.iloc[-1]
        ax.axvline(boundary, color="#95a5a6", linestyle="--", linewidth=1, alpha=0.7)

        # Prediksi
        ax.plot(
            forecast_dt, df_forecast[pred_col],
            color=color_pred, linewidth=2, linestyle="-",
            label=f"{label} (Prediksi 1 Jam)",
            marker="", markersize=0,
        )

        # Confidence band (visual, bukan statistik)
        pred_vals = df_forecast[pred_col].values
        ax.fill_between(
            forecast_dt,
            pred_vals * 0.85, pred_vals * 1.15,
            color=color_pred, alpha=0.1,
            label="Interval estimasi (+/- 15%)"
        )

        ax.set_ylabel(label, fontsize=10, color="#2c3e50")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        ax.grid(axis="both", alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=8, colors="#7f8c8d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title(
        f"Prediksi 1 Jam ke Depan  |  {model_name}  |  {node_label}  |  Skenario {SCENARIO}",
        fontsize=12, fontweight="bold", color="#2c3e50", pad=12
    )
    axes[1].set_xlabel("Waktu", fontsize=10, color="#2c3e50")

    plt.tight_layout()
    path = f"{output_dir}/inference_1h_scenario_{SCENARIO}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#f7f6f2")
    plt.close()
    return path


def run_inference(
    df: pd.DataFrame,
    results: dict,
    mode: str,
    features_used: list,
    output_dir: str = "outputs",
) -> dict:
    """
    Jalankan inference model terbaik untuk 1 jam ke depan.
    
    Parameters
    ----------
    df           : DataFrame hasil preprocessing (semua node)
    results      : dict hasil training (dari run_training)
    mode         : 'per_node' atau 'global'
    features_used: list nama fitur yang digunakan model
    output_dir   : direktori output
    
    Returns
    -------
    dict berisi:
        model_name : nama model pemenang
        forecast_df: DataFrame prediksi
        csv_path   : path file CSV
        plot_path  : path file plot
    """
    print(f"\n{'='*60}")
    print("  FASE 4: INFERENCE MODEL 1 JAM KE DEPAN")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    best_name, best_model = _get_best_model(results, mode)
    print(f"[Inference] Model pemenang: {best_name}")
    print(f"[Inference] Horizon: {FORECAST_HORIZON_SECONDS}s = {FORECAST_HORIZON_SECONDS // 60} menit")
    print(f"[Inference] Interval: setiap {DATA_INTERVAL_SECONDS}s | Total steps: {N_FORECAST_STEPS}")

    all_forecasts = []

    if mode == "per_node":
        for node_id, node_df in df.groupby("node_id"):
            node_df = node_df.sort_values("datetime").reset_index(drop=True)
            last_dt = pd.to_datetime(node_df["datetime"].iloc[-1])
            print(f"\n[Inference] Node {node_id} | Data terakhir: {last_dt}")

            # Ambil model per node
            node_model = results[node_id][best_name]["model"]

            forecast_df = _build_future_features(
                node_df, features_used, N_FORECAST_STEPS, node_model, mode
            )
            forecast_df["node_id"] = node_id

            # Plot per node
            plot_path = _plot_inference(
                node_df, forecast_df, best_name,
                node_label=f"Node {node_id}",
                output_dir=f"{output_dir}/plots",
            )
            # Rename plot per node
            node_plot = plot_path.replace(
                f"inference_1h_scenario_{SCENARIO}.png",
                f"inference_1h_node{node_id}_scenario_{SCENARIO}.png"
            )
            if plot_path != node_plot:
                os.rename(plot_path, node_plot)
                plot_path = node_plot

            print(f"[Inference] Plot tersimpan: {plot_path}")
            all_forecasts.append(forecast_df)
    else:
        # Global model: prediksi per node menggunakan model global yang sama
        for node_id, node_df in df.groupby("node_id"):
            node_df = node_df.sort_values("datetime").reset_index(drop=True)
            last_dt = pd.to_datetime(node_df["datetime"].iloc[-1])
            print(f"\n[Inference] Node {node_id} (model global) | Data terakhir: {last_dt}")

            forecast_df = _build_future_features(
                node_df, features_used, N_FORECAST_STEPS, best_model, mode
            )
            forecast_df["node_id"] = node_id
            all_forecasts.append(forecast_df)

        # Plot gabungan (gunakan node pertama sebagai representasi utama)
        first_node = list(df.groupby("node_id"))[0]
        node_id_first, node_df_first = first_node
        plot_path = _plot_inference(
            node_df_first, all_forecasts[0], best_name,
            node_label=f"Node {node_id_first} (Global)",
            output_dir=f"{output_dir}/plots",
        )
        print(f"[Inference] Plot tersimpan: {plot_path}")

    # Gabungkan semua hasil
    combined_df = pd.concat(all_forecasts, ignore_index=True)

    # Simpan CSV
    csv_path = f"{output_dir}/inference_1h_scenario_{SCENARIO}.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"\n[Inference] CSV prediksi tersimpan: {csv_path}")
    print(f"[Inference] Total baris prediksi: {len(combined_df)}")

    # Ringkasan statistik
    print(f"\n{'-'*60}")
    print("  RINGKASAN INFERENCE")
    print(f"{'-'*60}")
    for nid, grp in combined_df.groupby("node_id"):
        so2_mean = grp["so2_pred"].mean()
        h2s_mean = grp["h2s_pred"].mean()
        so2_max = grp["so2_pred"].max()
        h2s_max = grp["h2s_pred"].max()
        dt_start = grp["datetime"].iloc[0]
        dt_end = grp["datetime"].iloc[-1]
        print(f"  Node {nid}: {dt_start} -> {dt_end}")
        print(f"    SO2: mean={so2_mean:.2f}  max={so2_max:.2f} ug/m3")
        print(f"    H2S: mean={h2s_mean:.2f}  max={h2s_max:.2f} ug/m3")
    print(f"{'-'*60}\n")

    return {
        "model_name": best_name,
        "forecast_df": combined_df,
        "csv_path": csv_path,
        "plot_path": plot_path if mode != "per_node" else f"{output_dir}/plots/",
    }

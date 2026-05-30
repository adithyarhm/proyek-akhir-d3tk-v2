"""
FEATURE IMPORTANCE & SHAP ANALYSIS
====================================
Modul ini mengekstrak dan memvisualisasikan feature importance dari model
pemenang (model dengan RMSE terendah) hasil pipeline training.

Dua metode analisis yang didukung:
  1. Built-in Feature Importance  — langsung dari atribut model tree-based
     (feature_importances_ untuk RF/XGB/LightGBM, get_feature_importance()
     untuk CatBoost). Diplot sebagai grouped bar chart per target (SO2, H2S).
  2. SHAP (SHapley Additive exPlanations) — jika `shap` tersedia, digunakan
     TreeExplainer untuk analisis nilai kontribusi per fitur yang lebih
     interpretatif secara domain fisik.

Output visualisasi disimpan ke: outputs/plots/
  - feature_importance_builtin_<model>_<scenario>.png
  - shap_beeswarm_<model>_<target>_<scenario>.png   (jika shap tersedia)
  - shap_bar_<model>_<target>_<scenario>.png         (jika shap tersedia)

Cara penggunaan (dari main.py setelah run_training):
    from src.evaluation.feature_importance import run_feature_importance
    run_feature_importance(results, features_used, mode)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Gunakan backend non-interaktif agar aman dijalankan tanpa display
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.config import SCENARIO, TARGET_COL

# ─── Konstanta tampilan ──────────────────────────────────────────────────────
TARGET_LABELS = {
    "so2_ugm": "SO₂ (µg/m³)",
    "h2s_ugm": "H₂S (µg/m³)",
}
DOMAIN_HIGHLIGHT = {
    "temp_c":        "#e07b39",   # oranye — suhu
    "hum_pct":       "#4f9fbf",   # biru  — kelembapan
    "wind_kph":      "#5ca65c",   # hijau — angin
}
ACCENT_COLOR      = "#01696f"   # Hydra Teal (selaras gaya Nexus)
BAR_COLOR_DEFAULT = "#8fb8d0"
HIGHLIGHT_ALPHA   = 1.0
DEFAULT_ALPHA     = 0.75
FIG_DPI           = 160
FONT_BODY         = "DejaVu Sans"
OUTPUT_DIR        = "outputs/plots"


# ─── Utilitas ────────────────────────────────────────────────────────────────

def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _get_bar_colors(feature_names: list) -> list:
    """
    Kembalikan list warna per fitur.
    Fitur domain fisik (temp_c, hum_pct, wind_kph) diberi warna khusus;
    fitur lain menggunakan warna default.
    """
    return [
        DOMAIN_HIGHLIGHT.get(f, BAR_COLOR_DEFAULT)
        for f in feature_names
    ]


def _build_legend_patches() -> list:
    """Legend patch untuk fitur domain fisik yang di-highlight."""
    patches = [
        mpatches.Patch(color=DOMAIN_HIGHLIGHT["temp_c"],   label="Suhu Udara (temp_c)"),
        mpatches.Patch(color=DOMAIN_HIGHLIGHT["hum_pct"],  label="Kelembapan (hum_pct)"),
        mpatches.Patch(color=DOMAIN_HIGHLIGHT["wind_kph"], label="Kecepatan Angin (wind_kph)"),
        mpatches.Patch(color=BAR_COLOR_DEFAULT,            label="Fitur Lainnya"),
    ]
    return patches


def _select_winner_model(results: dict, mode: str) -> tuple[str, object, str | None]:
    """
    Pilih model pemenang berdasarkan RMSE terendah dari hasil training.

    Returns
    -------
    (model_name, model_object, node_id_or_none)
    """
    best_name = None
    best_rmse = float("inf")
    best_model = None
    best_node = None

    if mode == "per_node":
        # Hitung rata-rata RMSE tiap model di seluruh node
        model_names = set()
        for node_res in results.values():
            model_names.update(node_res.keys())

        avg_rmse: dict[str, float] = {}
        for m in model_names:
            rmses = [
                results[n][m]["metrics"]["rmse"]
                for n in results
                if m in results[n]
            ]
            avg_rmse[m] = sum(rmses) / len(rmses)

        best_name = min(avg_rmse, key=avg_rmse.get)
        best_rmse = avg_rmse[best_name]

        # Ambil model dari node pertama (representatif)
        first_node = next(iter(results))
        best_model = results[first_node][best_name]["model"]
        best_node = first_node

    elif mode == "global":
        for m_name, m_data in results["global"].items():
            rmse = m_data["metrics_overall"]["rmse"]
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = m_name
                best_model = m_data["model"]

    print(
        f"[FeatureImportance] Model pemenang: {best_name} "
        f"| RMSE: {best_rmse:.4f} | Mode: {mode}"
    )
    return best_name, best_model, best_node


def _extract_builtin_importance(
    model, features: list, model_name: str
) -> np.ndarray:
    """
    Ekstrak feature importance bawaan model.

    Mendukung:
      - RandomForest / XGBoost / LightGBM : .feature_importances_
      - CatBoost                           : .get_feature_importance()

    Returns
    -------
    numpy array shape (n_targets, n_features) — jika multi-output model
    mengembalikan importance per estimator, dijumlahkan lintas estimator;
    atau shape (n_features,) jika model menyimpan importance datar.
    """
    def _get_single_importance(est):
        if model_name == "CatBoost":
            try:
                return np.array(est.get_feature_importance())
            except Exception:
                pass
        return np.array(est.feature_importances_)

    if hasattr(model, "estimators_"):
        # Model multi-output (MultiOutputRegressor) yang membungkus beberapa estimator
        imp_list = []
        for est in model.estimators_:
            imp_list.append(_get_single_importance(est))
        return np.array(imp_list)
    else:
        return _get_single_importance(model)



# ─── Plot 1: Built-in Feature Importance ────────────────────────────────────

def plot_builtin_importance(
    model,
    model_name: str,
    features: list,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Buat dan simpan grouped bar chart feature importance bawaan model.

    Jika model mengembalikan importance 2-D (satu baris per target), setiap
    target (SO₂ dan H₂S) divisualisasikan berdampingan.  Jika importance
    berbentuk 1-D (dirangkum lintas target), ditampilkan satu panel.

    Parameters
    ----------
    model       : model tree-based yang sudah dilatih
    model_name  : nama model string (untuk judul dan nama file)
    features    : list nama fitur yang digunakan saat training
    output_dir  : direktori output

    Returns
    -------
    str — path file gambar yang tersimpan
    """
    _ensure_output_dir()
    imp_raw = _extract_builtin_importance(model, features, model_name)

    n_features = len(features)

    # ── Normalisasi shape ────────────────────────────────────────────────────
    if imp_raw.ndim == 2 and imp_raw.shape[0] == len(TARGET_COL):
        # shape (n_targets, n_features) → susun per target
        imps = {TARGET_COL[i]: imp_raw[i] for i in range(len(TARGET_COL))}
    elif imp_raw.ndim == 2 and imp_raw.shape[1] == n_features:
        # shape (n_estimators, n_features) → rata-rata estimator
        imp_mean = imp_raw.mean(axis=0)
        imps = {"combined": imp_mean}
    else:
        # shape (n_features,)
        imps = {"combined": imp_raw}

    n_panels = len(imps)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(max(10, 5 * n_panels + 2), 7),
        facecolor="#f7f6f2"
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (target_key, imp) in zip(axes, imps.items()):
        # Urutkan dari terbesar ke terkecil
        order = np.argsort(imp)[::-1]
        sorted_features = [features[i] for i in order]
        sorted_imp = imp[order]

        colors = _get_bar_colors(sorted_features)
        bars = ax.barh(
            sorted_features[::-1],    # balik agar terbesar di atas
            sorted_imp[::-1],
            color=colors[::-1],
            edgecolor="white",
            linewidth=0.6,
            alpha=DEFAULT_ALPHA,
        )

        # Anotasi nilai di ujung bar
        for bar, val in zip(bars, sorted_imp[::-1]):
            ax.text(
                bar.get_width() + max(sorted_imp) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left",
                fontsize=8, color="#28251d"
            )

        target_label = (
            TARGET_LABELS.get(target_key, target_key.upper())
            if target_key != "combined"
            else "Rata-rata Lintas Target"
        )
        ax.set_title(
            f"Feature Importance — {target_label}",
            fontsize=11, fontweight="bold", color="#28251d", pad=10
        )
        ax.set_xlabel("Importance Score", fontsize=9, color="#7a7974")
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8, colors="#7a7974")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#dcd9d5")
        ax.spines["bottom"].set_color("#dcd9d5")
        ax.set_facecolor("#f9f8f5")
        ax.grid(axis="x", alpha=0.3, linestyle="--", color="#dcd9d5")

    # Judul utama
    fig.suptitle(
        f"Built-in Feature Importance — Model: {model_name}  |  Skenario {SCENARIO}",
        fontsize=13, fontweight="bold", color="#28251d", y=1.02
    )

    # Legend domain fisik
    fig.legend(
        handles=_build_legend_patches(),
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        frameon=True,
        facecolor="#f9f8f5",
        edgecolor="#dcd9d5",
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f"feature_importance_builtin_{model_name.lower()}_scenario{SCENARIO}.png"
    )
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="#f7f6f2")
    plt.close()
    print(f"[FeatureImportance] Saved: {path}")
    return path


# ─── Plot 2: SHAP Analysis ───────────────────────────────────────────────────

def plot_shap_analysis(
    model,
    model_name: str,
    X_test: np.ndarray,
    features: list,
    output_dir: str = OUTPUT_DIR,
    max_display: int = 15,
) -> list[str]:
    """
    Jalankan SHAP TreeExplainer dan simpan:
      - Beeswarm plot (distribusi kontribusi per fitur per sampel) untuk setiap target
      - Bar plot SHAP (nilai absolut rata-rata) untuk setiap target

    Parameters
    ----------
    model       : model tree-based yang sudah dilatih
    model_name  : nama model string
    X_test      : data uji sebagai numpy array (n_samples, n_features)
    features    : list nama fitur
    output_dir  : direktori output
    max_display : jumlah fitur teratas yang ditampilkan di SHAP plot

    Returns
    -------
    list[str] — list path file gambar yang tersimpan
    """
    if not SHAP_AVAILABLE:
        print("[FeatureImportance] SHAP tidak tersedia. Lewati analisis SHAP.")
        print("  → Instal dengan: pip install shap")
        return []

    _ensure_output_dir()
    saved_paths = []

    # Batasi sampel untuk efisiensi komputasi (maks 500 sampel)
    n_samples = min(500, X_test.shape[0])
    X_sample = X_test[:n_samples]

    print(
        f"[FeatureImportance] Menjalankan SHAP TreeExplainer pada {n_samples} sampel..."
    )

    try:
        if hasattr(model, "estimators_"):
            # Multi-output model (MultiOutputRegressor), hitung SHAP per estimator
            shap_values = []
            for est in model.estimators_:
                explainer = shap.TreeExplainer(est)
                sv = explainer.shap_values(X_sample)
                # Di beberapa versi SHAP/model, output shap_values bisa berupa list atau 3D array
                if isinstance(sv, list):
                    sv = sv[0]
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    sv = sv[:, :, 0]
                shap_values.append(sv)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"[FeatureImportance] SHAP gagal dijalankan: {e}")
        return []

    # Normalisasi shap_values ke list of arrays
    # Multi-output: shap_values bisa berupa list[array] (satu per target)
    # Single-output: shap_values adalah array tunggal
    if isinstance(shap_values, list):
        targets_shap = {TARGET_COL[i]: shap_values[i] for i in range(len(shap_values))}
    else:
        # Jika model menyimpan shap values 3-D: (samples, features, targets)
        if shap_values.ndim == 3:
            targets_shap = {
                TARGET_COL[i]: shap_values[:, :, i]
                for i in range(shap_values.shape[2])
            }
        else:
            targets_shap = {"combined": shap_values}

    for target_key, sv in targets_shap.items():
        target_label = TARGET_LABELS.get(target_key, target_key.upper())

        # ── Beeswarm plot ────────────────────────────────────────────────────
        fig_bee, ax_bee = plt.subplots(figsize=(10, 6), facecolor="#f7f6f2")
        shap.summary_plot(
            sv, X_sample,
            feature_names=features,
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        plt.title(
            f"SHAP Beeswarm — {target_label}  |  {model_name}  |  Skenario {SCENARIO}",
            fontsize=11, fontweight="bold", color="#28251d", pad=10
        )
        plt.tight_layout()
        path_bee = os.path.join(
            output_dir,
            f"shap_beeswarm_{model_name.lower()}_{target_key}_scenario{SCENARIO}.png"
        )
        plt.savefig(path_bee, dpi=FIG_DPI, bbox_inches="tight", facecolor="#f7f6f2")
        plt.close()
        print(f"[FeatureImportance] Saved: {path_bee}")
        saved_paths.append(path_bee)

        # ── SHAP Bar plot (mean |SHAP|) ───────────────────────────────────────
        # Hitung mean absolute SHAP per fitur secara manual untuk kontrol penuh
        mean_abs_shap = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs_shap)[::-1][:max_display]
        sorted_feats = [features[i] for i in order]
        sorted_vals  = mean_abs_shap[order]

        fig_bar, ax_bar = plt.subplots(figsize=(9, 6), facecolor="#f7f6f2")
        colors = _get_bar_colors(sorted_feats)
        ax_bar.barh(
            sorted_feats[::-1],
            sorted_vals[::-1],
            color=colors[::-1],
            edgecolor="white",
            linewidth=0.6,
            alpha=DEFAULT_ALPHA,
        )
        ax_bar.set_title(
            f"SHAP Mean |Value| — {target_label}  |  {model_name}  |  Skenario {SCENARIO}",
            fontsize=11, fontweight="bold", color="#28251d", pad=10
        )
        ax_bar.set_xlabel("mean(|SHAP value|)", fontsize=9, color="#7a7974")
        ax_bar.tick_params(axis="y", labelsize=9)
        ax_bar.tick_params(axis="x", labelsize=8, colors="#7a7974")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_color("#dcd9d5")
        ax_bar.spines["bottom"].set_color("#dcd9d5")
        ax_bar.set_facecolor("#f9f8f5")
        ax_bar.grid(axis="x", alpha=0.3, linestyle="--", color="#dcd9d5")
        ax_bar.legend(
            handles=_build_legend_patches(),
            fontsize=8, frameon=True,
            facecolor="#f9f8f5", edgecolor="#dcd9d5",
        )
        plt.tight_layout()
        path_bar = os.path.join(
            output_dir,
            f"shap_bar_{model_name.lower()}_{target_key}_scenario{SCENARIO}.png"
        )
        plt.savefig(path_bar, dpi=FIG_DPI, bbox_inches="tight", facecolor="#f7f6f2")
        plt.close()
        print(f"[FeatureImportance] Saved: {path_bar}")
        saved_paths.append(path_bar)

    return saved_paths


# ─── Plot 3: Domain Analysis — Suhu, Kelembapan, Angin ──────────────────────

def plot_domain_analysis(
    model,
    model_name: str,
    features: list,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Visualisasi khusus domain fisik: tampilkan kontribusi relatif tiga fitur
    utama (temp_c, hum_pct, wind_kph) dibandingkan fitur lainnya dalam
    bentuk pie chart dan detail bar chart side-by-side.

    Menggunakan built-in importance sebagai dasar analisis.

    Returns
    -------
    str — path file gambar yang tersimpan, atau "" jika fitur domain tidak ada
    """
    _ensure_output_dir()
    imp_raw = _extract_builtin_importance(model, features, model_name)

    # Gunakan importance total (rata-rata lintas target jika 2-D)
    if imp_raw.ndim == 2:
        imp_1d = imp_raw.mean(axis=0)
    else:
        imp_1d = imp_raw

    if imp_1d.shape[0] != len(features):
        print(
            "[FeatureImportance] Dimensi importance tidak cocok dengan features. "
            "Lewati domain analysis."
        )
        return ""

    domain_keys   = ["temp_c", "hum_pct", "wind_kph"]
    domain_labels = ["Suhu Udara\n(temp_c)", "Kelembapan\n(hum_pct)", "Kec. Angin\n(wind_kph)"]
    domain_colors = [DOMAIN_HIGHLIGHT["temp_c"], DOMAIN_HIGHLIGHT["hum_pct"], DOMAIN_HIGHLIGHT["wind_kph"]]

    present_keys = [k for k in domain_keys if k in features]
    if not present_keys:
        print("[FeatureImportance] Tidak ada fitur domain fisik di features list. Lewati domain analysis.")
        return ""

    feat_idx     = {f: i for i, f in enumerate(features)}
    domain_imp   = {k: imp_1d[feat_idx[k]] for k in present_keys}
    other_imp    = sum(
        imp_1d[i] for i, f in enumerate(features)
        if f not in domain_keys
    )

    # ── Layout: pie + bar berdampingan ────────────────────────────────────────
    fig, (ax_pie, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 6), facecolor="#f7f6f2"
    )

    # Pie chart proporsi domain vs non-domain
    pie_labels = [DOMAIN_HIGHLIGHT.get(k, k) for k in present_keys] + ["Fitur Lainnya"]
    pie_vals   = list(domain_imp.values()) + [other_imp]
    pie_colors = [domain_colors[domain_keys.index(k)] for k in present_keys] + [BAR_COLOR_DEFAULT]
    pie_explode = [0.04] * len(present_keys) + [0.0]

    wedges, texts, autotexts = ax_pie.pie(
        pie_vals,
        labels=[
            ("Suhu Udara\n(temp_c)" if k == "temp_c"
             else "Kelembapan\n(hum_pct)" if k == "hum_pct"
             else "Kec. Angin\n(wind_kph)" if k == "wind_kph"
             else "Fitur Lainnya")
            for k in (present_keys + ["other"])
        ],
        colors=pie_colors,
        explode=pie_explode,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 9, "color": "#28251d"},
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_color("#28251d")
    ax_pie.set_title(
        "Proporsi Kontribusi Fitur Domain Fisik",
        fontsize=11, fontweight="bold", color="#28251d", pad=12
    )
    ax_pie.set_facecolor("#f9f8f5")

    # Bar chart detail nilai importance fitur domain
    bar_labels = [
        "Suhu Udara\n(temp_c)" if k == "temp_c"
        else "Kelembapan\n(hum_pct)" if k == "hum_pct"
        else "Kec. Angin\n(wind_kph)"
        for k in present_keys
    ]
    bar_vals   = [domain_imp[k] for k in present_keys]
    bar_colors = [DOMAIN_HIGHLIGHT[k] for k in present_keys]

    bars = ax_bar.bar(
        bar_labels, bar_vals,
        color=bar_colors, edgecolor="white", linewidth=1.0, alpha=DEFAULT_ALPHA
    )
    for bar, val in zip(bars, bar_vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bar_vals) * 0.02,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9.5, color="#28251d", fontweight="bold"
        )

    ax_bar.set_ylabel("Feature Importance Score", fontsize=9, color="#7a7974")
    ax_bar.set_title(
        "Importance Fitur Domain Fisik Kawah Putih",
        fontsize=11, fontweight="bold", color="#28251d", pad=12
    )
    ax_bar.tick_params(axis="both", labelsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color("#dcd9d5")
    ax_bar.spines["bottom"].set_color("#dcd9d5")
    ax_bar.set_facecolor("#f9f8f5")
    ax_bar.grid(axis="y", alpha=0.3, linestyle="--", color="#dcd9d5")

    fig.suptitle(
        f"Analisis Domain Fisik — {model_name}  |  Skenario {SCENARIO}\n"
        f"Pengaruh Suhu, Kelembapan & Angin terhadap Prediksi Gas SO₂/H₂S — Kawah Putih",
        fontsize=12, fontweight="bold", color="#28251d", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f"feature_importance_domain_{model_name.lower()}_scenario{SCENARIO}.png"
    )
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="#f7f6f2")
    plt.close()
    print(f"[FeatureImportance] Saved: {path}")
    return path


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run_feature_importance(
    results: dict,
    features_used: list,
    mode: str,
    X_test: np.ndarray | None = None,
    use_shap: bool = True,
) -> dict:
    """
    Fungsi utama — ekstrak feature importance dari model pemenang dan simpan
    semua visualisasi ke `outputs/plots/`.

    Parameters
    ----------
    results      : dict hasil run_training() — format sama persis dengan
                   output train_per_node() atau train_global()
    features_used: list nama fitur yang dipakai saat training (urutan harus
                   sama dengan urutan kolom di X_train/X_test)
    mode         : "per_node" atau "global"
    X_test       : (opsional) numpy array data uji untuk analisis SHAP.
                   Jika None, SHAP dilewati meski paket shap tersedia.
    use_shap     : flag untuk mengaktifkan/menonaktifkan analisis SHAP.

    Returns
    -------
    dict berisi:
        "winner_model_name" : str
        "winner_model"      : model object
        "builtin_plot"      : str path gambar built-in importance
        "domain_plot"       : str path gambar domain analysis
        "shap_plots"        : list[str] path gambar SHAP (kosong jika dilewati)
    """
    print(f"\n{'='*60}")
    print("  FASE 3: FEATURE IMPORTANCE & DOMAIN ANALYSIS")
    print(f"{'='*60}")

    model_name, model, node_id = _select_winner_model(results, mode)

    if model is None:
        print("[FeatureImportance] ERROR: model tidak ditemukan di results.")
        return {}

    # ── Plot 1: Built-in importance ──────────────────────────────────────────
    builtin_path = plot_builtin_importance(model, model_name, features_used)

    # ── Plot 2: Domain analysis ──────────────────────────────────────────────
    domain_path = plot_domain_analysis(model, model_name, features_used)

    # ── Plot 3: SHAP (jika diminta dan data tersedia) ─────────────────────────
    shap_paths: list[str] = []
    if use_shap and X_test is not None:
        shap_paths = plot_shap_analysis(model, model_name, X_test, features_used)
    elif use_shap and X_test is None:
        print(
            "[FeatureImportance] X_test tidak diberikan → SHAP dilewati.\n"
            "  → Untuk mengaktifkan SHAP, berikan argumen X_test ke run_feature_importance()."
        )

    # ── Ringkasan ─────────────────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"  RINGKASAN FASE 3")
    print(f"{'-'*60}")
    print(f"  Model Pemenang   : {model_name}")
    print(f"  Built-in plot    : {builtin_path}")
    print(f"  Domain plot      : {domain_path}")
    if shap_paths:
        print(f"  SHAP plots ({len(shap_paths)})  : {shap_paths[0]} ...")
    else:
        shap_status = "SHAP tersedia (butuh X_test)" if SHAP_AVAILABLE else "shap tidak terinstal"
        print(f"  SHAP plots       : Dilewati ({shap_status})")
    print(f"{'-'*60}\n")

    return {
        "winner_model_name": model_name,
        "winner_model":      model,
        "builtin_plot":      builtin_path,
        "domain_plot":       domain_path,
        "shap_plots":        shap_paths,
    }

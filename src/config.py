"""
CONFIGURATION SETTINGS
======================
Pusat pengaturan untuk seluruh pipeline ML, termasuk pemilihan skenario,
definisi fitur, path data, dan parameter pelatihan.
"""

# Pilih skenario (1-4):
#   1 = Per-Node,   fitur dasar saja
#   2 = Per-Node,   fitur dasar + temporal + turunan
#   3 = Global,     fitur dasar saja
#   4 = Global,     fitur dasar + temporal + turunan

SCENARIO = 1  # <-- UBAH SKENARIO DI SINI

TARGET_COL = ["so2_ugm", "h2s_ugm"]
BASE_FEATURES = ["hum_pct", "temp_c", "wind_kph"]
TEMPORAL_FEATURES = ["hour", "minute", "minute_of_day"]
DERIVED_FEATURES = ["h2s_diff", "so2_diff", "gas_ratio_so2_h2s",
                    "so2_ugm_lag1", "so2_ugm_lag2", "h2s_ugm_lag1",
                    "h2s_ugm_lag2"]
SPATIAL_FEATURES = ["lat", "lon", "elev"]
ALL_FEATURES = BASE_FEATURES + TEMPORAL_FEATURES + DERIVED_FEATURES + SPATIAL_FEATURES + TARGET_COL

SCENARIO_CONFIG = {
    1: {
        "name": "Baseline Per-Node",
        "mode": "per_node",
        "features": BASE_FEATURES,
    },
    2: {
        "name": "Enhanced Per-Node",
        "mode": "per_node",
        "features": BASE_FEATURES + TEMPORAL_FEATURES + DERIVED_FEATURES,
    },
    3: {
        "name": "Global Model (Baseline)",
        "mode": "global",
        "features": BASE_FEATURES,
    },
    4: {
        "name": "Global Model + Temporal",
        "mode": "global",
        "features": BASE_FEATURES + TEMPORAL_FEATURES + DERIVED_FEATURES,
    },
}



# DATA PATHS & DIRECTORIES
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
DATA_PATH = "data/processed/node_combined_final.csv"

# Model Saving & Logs
MODEL_SAVE_DIR = "saved-models"
LOG_FILE = "logs/"
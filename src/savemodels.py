"""
SAVE & LOAD MODELS
==================
Serialisasi model terbaik menggunakan joblib.
"""

import os
import joblib
from src.config import MODEL_SAVE_DIR


def save_model(model, model_name: str, node_id: str = "global"):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    fname = f"{MODEL_SAVE_DIR}/{model_name}_{node_id}.pkl"
    joblib.dump(model, fname)
    print(f"[SaveModel] Tersimpan: {fname}")
    return fname


def load_model(model_name: str, node_id: str = "global"):
    fname = f"{MODEL_SAVE_DIR}/{model_name}_{node_id}.pkl"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Model tidak ditemukan: {fname}")
    return joblib.load(fname)


def save_all_models(results: dict, mode: str):
    """Simpan semua model dari hasil training."""
    if mode == "per_node":
        for node_id, node_res in results.items():
            for mname, mdata in node_res.items():
                save_model(mdata["model"], mname, str(node_id))
    else:
        for mname, mdata in results["global"].items():
            save_model(mdata["model"], mname, "global")
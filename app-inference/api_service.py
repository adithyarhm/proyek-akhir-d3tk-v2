import os
import sys
import functools
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib

app = FastAPI(title="Kawah Putih Gas Prediction API", version="1.0.0")

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tentukan direktori model relatif terhadap lokasi file ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "saved-models")

# Mapping ID Node dari dashboard (N-01, N-02) ke file model (1, 2)
NODE_MAPPING = {
    "N-01": "1",
    "N-02": "2",
    "N-03": "1", # Fallback ke node 1 jika tidak ada model khusus N-03
    "N-04": "2", # Fallback ke node 2 jika tidak ada model khusus N-04
    "1": "1",
    "2": "2",
}

# Mapping nama model dari dashboard ke file model .pkl asli
MODEL_NAME_MAPPING = {
    "gradientboosting": "LightGBM",
    "gradientboost": "LightGBM",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "randomforest": "RandomForest",
    "lightgbm": "LightGBM"
}

class PredictionRequest(BaseModel):
    scenario: int
    model_name: str
    node_id: str
    temp_c: float
    hum_pct: float
    wind_kph: float
    hour: Optional[int] = 0
    minute: Optional[int] = 0
    minute_of_day: Optional[int] = 0
    h2s_diff: Optional[float] = 0.0
    so2_diff: Optional[float] = 0.0
    gas_ratio_so2_h2s: Optional[float] = 0.0
    so2_ugm_lag1: Optional[float] = 0.0
    so2_ugm_lag2: Optional[float] = 0.0
    h2s_ugm_lag1: Optional[float] = 0.0
    h2s_ugm_lag2: Optional[float] = 0.0
    lat: Optional[float] = 0.0
    lon: Optional[float] = 0.0
    elev: Optional[float] = 0.0

@functools.lru_cache(maxsize=32)
def load_model_cached(model_path: str):
    """Memuat model dari disk dan menyimpannya di memory cache."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di path: {model_path}")
    return joblib.load(model_path)

@app.get("/status")
def status():
    return {
        "status": "online",
        "model_dir": os.path.abspath(MODEL_DIR),
        "available_models": os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
    }

@app.post("/predict")
def predict(req: PredictionRequest):
    # 1. Tentukan nama file model berdasarkan skenario
    # Skenario 1 & 2: mode per_node (RandomForest_1.pkl, XGBoost_2.pkl, dll)
    # Skenario 3 & 4: mode global (RandomForest_global.pkl, XGBoost_global.pkl, dll)
    normalized_model = MODEL_NAME_MAPPING.get(req.model_name.lower(), req.model_name)
    if req.scenario in [1, 2]:
        node_key = NODE_MAPPING.get(req.node_id, "1")
        model_filename = f"{normalized_model}_{node_key}.pkl"
    elif req.scenario in [3, 4]:
        model_filename = f"{normalized_model}_global.pkl"
    else:
        raise HTTPException(status_code=400, detail=f"Skenario {req.scenario} tidak valid (harus 1-4)")

    model_path = os.path.join(MODEL_DIR, model_filename)

    try:
        model = load_model_cached(model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memuat model: {str(e)}")

    # 2. Susun fitur ke dalam format array numpy sesuai skenario
    # Urutan fitur harus persis sama dengan yang digunakan saat training
    try:
        if req.scenario == 1:
            # Baseline Per-Node: hum_pct, temp_c, wind_kph
            features = [req.hum_pct, req.temp_c, req.wind_kph]
        elif req.scenario == 2:
            # Enhanced Per-Node: hum_pct, temp_c, wind_kph, hour, minute, minute_of_day, h2s_diff, so2_diff, gas_ratio_so2_h2s, so2_ugm_lag1, so2_ugm_lag2, h2s_ugm_lag1, h2s_ugm_lag2
            features = [
                req.hum_pct, req.temp_c, req.wind_kph,
                req.hour, req.minute, req.minute_of_day,
                req.h2s_diff, req.so2_diff, req.gas_ratio_so2_h2s,
                req.so2_ugm_lag1, req.so2_ugm_lag2,
                req.h2s_ugm_lag1, req.h2s_ugm_lag2
            ]
        elif req.scenario == 3:
            # Global Model (Baseline): hum_pct, temp_c, wind_kph, lat, lon, elev
            features = [
                req.hum_pct, req.temp_c, req.wind_kph,
                req.lat, req.lon, req.elev
            ]
        elif req.scenario == 4:
            # Global Model + Temporal: hum_pct, temp_c, wind_kph, hour, minute, minute_of_day, h2s_diff, so2_diff, gas_ratio_so2_h2s, so2_ugm_lag1, so2_ugm_lag2, h2s_ugm_lag1, h2s_ugm_lag2, lat, lon, elev
            features = [
                req.hum_pct, req.temp_c, req.wind_kph,
                req.hour, req.minute, req.minute_of_day,
                req.h2s_diff, req.so2_diff, req.gas_ratio_so2_h2s,
                req.so2_ugm_lag1, req.so2_ugm_lag2,
                req.h2s_ugm_lag1, req.h2s_ugm_lag2,
                req.lat, req.lon, req.elev
            ]
        else:
            raise HTTPException(status_code=400, detail="Urutan fitur skenario tidak terdefinisi")

        # 3. Lakukan inferensi (mengembalikan array [so2_ugm, h2s_ugm])
        features_array = np.array([features])
        prediction = model.predict(features_array)

        # MultiOutputRegressor menghasilkan array 2D: [[so2, h2s]]
        so2_pred = float(prediction[0][0])
        h2s_pred = float(prediction[0][1])

        return {
            "so2_pred": round(so2_pred, 4),
            "h2s_pred": round(h2s_pred, 4),
            "model_used": model_filename,
            "features_sent": features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan inferensi: {str(e)}")

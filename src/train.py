"""
TRAINING
===============
Melatih beberapa model regresi pada data peramalan gas yang telah diproses.
Mendukung dua mode pelatihan:
  - per_node : Setiap node mendapatkan model terpisah (Skenario 1 & 2)
  - global   : Satu model dilatih pada data gabungan semua node (Skenario 3 & 4)

Jika tuning_results disediakan, model yang sudah di-tuning (best_model)
akan digunakan langsung tanpa melatih ulang dari nol.

Evaluasi metrik selalu dilaporkan PER-NODE agar performa antar skenario
bisa dibandingkan secara adil.
"""

from src.models.catboost import catboost_model
from src.models.lgbm import light_gbm_model
from src.models.xgboost import xgboost_model
from src.models.rfr import random_forest_regressor_model
from src.evaluation.metrics import *
from src.config import *
from sklearn.model_selection import TimeSeriesSplit


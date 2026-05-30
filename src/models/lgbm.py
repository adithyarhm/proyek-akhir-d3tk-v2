from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from src.config import * 
    
def light_gbm_model(X_train, y_train):
    """
    Ini untuk Build dan train model LightGBM dengan MultiOutputRegressor
    """
    lgbm_model = LGBMRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    # Train LightGBM pakai MultiOutputRegressor
    lgbm = MultiOutputRegressor(lgbm_model)
    lgbm.fit(X_train, y_train)                  # training
    
    return lgbm                                 # balikin model
    
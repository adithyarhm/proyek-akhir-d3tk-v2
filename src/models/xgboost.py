from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from src.config import * 
    
def xgboost_model(X_train, y_train):
    """
    Ini untuk build model XGBoost
    """
    xgbr_model = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    # Train XGBoost pakai MultiOutputRegressor
    xgbr = MultiOutputRegressor(xgbr_model)
    xgbr.fit(X_train, y_train)                       # training
    
    return xgbr                                      # balikin model
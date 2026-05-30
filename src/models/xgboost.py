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
        n_estimators=550,
        learning_rate=0.05373075383602183,
        max_depth=8,
        subsample=0.8049771821079034,
        colsample_bytree=0.9731026804971433,
        min_child_weight=3,
        gamma=0.46721163657062076,
        reg_alpha=0.0941784778791576,
        reg_lambda=2.9274637258722955e-05
    )

    # Train XGBoost pakai MultiOutputRegressor
    xgbr = MultiOutputRegressor(xgbr_model)
    xgbr.fit(X_train, y_train)                       # training
    
    return xgbr                                      # balikin model
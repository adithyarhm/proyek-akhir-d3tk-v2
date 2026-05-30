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
        verbosity=-1,
        max_depth= 9,
        num_leaves= 132,
        n_estimators= 250,
        learning_rate= 0.029051719888849394,
        min_child_samples= 7,
        subsample= 0.5463604730071667,
        colsample_bytree= 0.9756199445705375,
        reg_alpha= 0.4016396576514225,
        reg_lambda= 0.12753665526385652
    )

    # Train LightGBM pakai MultiOutputRegressor
    lgbm = MultiOutputRegressor(lgbm_model)
    lgbm.fit(X_train, y_train)                  # training
    
    return lgbm                                 # balikin model
    
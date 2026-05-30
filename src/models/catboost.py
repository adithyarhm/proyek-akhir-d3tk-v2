from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
from src.config import * 
    
def catboost_model(X_train, y_train):
    """
    Ini untuk Build dan train model CatBoost dengan MultiOutputRegressor
    """
    cat_model = CatBoostRegressor(
        random_state=42,
        silent=True,
        allow_writing_files=False
    )

    # Train CatBoost pakai MultiOutputRegressor
    cat = MultiOutputRegressor(cat_model)
    cat.fit(X_train, y_train)                  # training
    
    return cat                                 # balikin model
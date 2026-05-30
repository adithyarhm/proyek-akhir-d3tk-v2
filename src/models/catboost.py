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
        allow_writing_files=False,
        iterations=450,
        learning_rate=0.21155741395841124,
        depth=5,
        l2_leaf_reg=4.998646486979096,
        random_strength=0.02487587506524233,
        bagging_temperature=0.5664049362748634,
        border_count=203
    )

    # Train CatBoost pakai MultiOutputRegressor
    cat = MultiOutputRegressor(cat_model)
    cat.fit(X_train, y_train)                  # training
    
    return cat                                 # balikin model
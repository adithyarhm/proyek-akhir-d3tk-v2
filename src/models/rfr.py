from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from src.config import * 
    
def random_forest_regressor_model(X_train, y_train):
    """
    Ini untuk build model Random Forest Regressor
    """
    rfr_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=0,
        n_estimators=250,
        max_depth=22,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="log2"
    )
    
    # Train Random Forest Regressor pakai MultiOutputRegressor
    rfr = MultiOutputRegressor(rfr_model)
    rfr.fit(X_train, y_train)                  # training

    return rfr                                 # balikin model
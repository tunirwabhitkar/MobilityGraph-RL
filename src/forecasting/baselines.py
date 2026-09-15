import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any

class NaiveBaseline:
    """Predicts next time step as the current time step."""
    def predict(self, df: pd.DataFrame, target_col: str = 'demand') -> np.ndarray:
        return df[target_col].values

class MovingAverageBaseline:
    """Predicts next time step as the moving average of the last `window` steps."""
    def __init__(self, window: int = 3):
        self.window = window
        
    def predict(self, df: pd.DataFrame, target_col: str = 'demand') -> np.ndarray:
        return df.groupby('zone_id')[target_col].shift(1).rolling(window=self.window).mean().values

class XGBoostBaseline:
    """XGBoost tabular model for demand forecasting."""
    def __init__(self, params: Dict[str, Any] = None):
        if params is None:
            self.params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
        else:
            self.params = params
        self.model = xgb.XGBRegressor(**self.params)
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

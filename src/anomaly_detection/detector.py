import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    """
    Detects demand anomalies using Z-Score and Isolation Forest on forecasting residuals.
    """
    def __init__(self, contamination: float = 0.05, z_threshold: float = 3.0):
        self.contamination = contamination
        self.z_threshold = z_threshold
        self.iforest = IsolationForest(contamination=self.contamination, random_state=42)
        
    def fit(self, residuals: np.ndarray):
        """Fits the Isolation Forest on training residuals."""
        # Reshape to 2D for scikit-learn
        self.iforest.fit(residuals.reshape(-1, 1))
        
    def detect(self, expected: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
        """
        Detects anomalies based on the gap between expected (predicted) and actual demand.
        Returns a DataFrame with anomaly scores and flags.
        """
        residuals = actual - expected
        
        # Z-score based detection
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        z_scores = (residuals - mean_res) / (std_res + 1e-8)
        z_anomaly = np.abs(z_scores) > self.z_threshold
        
        # Isolation Forest detection
        iforest_preds = self.iforest.predict(residuals.reshape(-1, 1))
        if_anomaly = iforest_preds == -1 # -1 means anomaly
        
        # Combine
        combined_anomaly = z_anomaly | if_anomaly
        
        df = pd.DataFrame({
            'expected_demand': expected,
            'actual_demand': actual,
            'residual': residuals,
            'z_score': z_scores,
            'is_anomaly_z': z_anomaly,
            'is_anomaly_if': if_anomaly,
            'is_anomaly_combined': combined_anomaly,
            'anomaly_score': np.abs(z_scores) # Simple mapping for score
        })
        
        return df

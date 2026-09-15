import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

class DriftMonitor:
    """Tracks model performance and feature distribution drift."""
    
    def calculate_mae(self, expected: np.ndarray, actual: np.ndarray) -> float:
        return np.mean(np.abs(expected - actual))
        
    def calculate_mape(self, expected: np.ndarray, actual: np.ndarray) -> float:
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - expected[mask]) / actual[mask])) * 100
        
    def detect_feature_drift(self, reference_data: np.ndarray, current_data: np.ndarray, alpha: float = 0.05) -> dict:
        """
        Uses Kolmogorov-Smirnov test to detect distribution drift on features.
        """
        num_features = reference_data.shape[1]
        drift_results = {}
        
        for i in range(num_features):
            ref_feat = reference_data[:, i]
            cur_feat = current_data[:, i]
            
            # KS Test
            stat, p_value = ks_2samp(ref_feat, cur_feat)
            
            is_drifting = p_value < alpha
            drift_results[f'feature_{i}'] = {
                'ks_stat': stat,
                'p_value': p_value,
                'is_drifting': is_drifting
            }
            
        return drift_results
        
    def generate_report(self, expected: np.ndarray, actual: np.ndarray, ref_features: np.ndarray, cur_features: np.ndarray) -> dict:
        mae = self.calculate_mae(expected, actual)
        mape = self.calculate_mape(expected, actual)
        drift = self.detect_feature_drift(ref_features, cur_features)
        
        any_drift = any(d['is_drifting'] for d in drift.values())
        
        return {
            'Performance': {
                'MAE': mae,
                'MAPE (%)': mape
            },
            'Drift Detected': any_drift,
            'Feature Drift Details': drift
        }

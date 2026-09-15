import pandas as pd
import numpy as np

class ImbalanceEngine:
    """Calculates demand-supply gap and classifies zone status."""
    
    def calculate_gap(self, predictions: np.ndarray, available_supply: np.ndarray, zone_ids: np.ndarray) -> pd.DataFrame:
        """
        predictions: Expected demand per zone [num_zones]
        available_supply: Available vehicles per zone [num_zones]
        zone_ids: IDs of the zones [num_zones]
        """
        gap = predictions - available_supply
        
        df = pd.DataFrame({
            'zone_id': zone_ids,
            'predicted_demand': predictions,
            'available_vehicles': available_supply,
            'demand_gap': gap
        })
        
        # Classify status
        # Gap > 0 means shortage (demand > supply)
        # Gap < 0 means surplus (supply > demand)
        conditions = [
            (df['demand_gap'] >= 20),
            (df['demand_gap'] > 5) & (df['demand_gap'] < 20),
            (df['demand_gap'] >= -5) & (df['demand_gap'] <= 5),
            (df['demand_gap'] < -5)
        ]
        choices = ['CRITICAL SHORTAGE', 'SHORTAGE', 'BALANCED', 'SURPLUS']
        
        df['status'] = np.select(conditions, choices, default='UNKNOWN')
        
        return df

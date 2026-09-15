import pandas as pd

class FeatureEngineer:
    """
    Adds lag and rolling features to the time series dataset.
    Needs to operate per zone to prevent spatial leakage.
    """
    def __init__(self, lag_hours: list = [1, 2, 24, 168], rolling_windows: list = [3, 6, 24]):
        self.lag_hours = lag_hours
        self.rolling_windows = rolling_windows
        
    def engineer_features(self, df: pd.DataFrame, target_col: str = 'demand') -> pd.DataFrame:
        """
        Calculates features. Assumes df is sorted chronologically for each zone.
        """
        # Ensure correct sorting
        df = df.sort_values(by=['zone_id', 'timestamp'])
        
        features_dfs = []
        
        # We group by zone_id to compute time-series features
        for zone_id, zone_df in df.groupby('zone_id'):
            zone_df = zone_df.copy()
            
            # Lag features
            for lag in self.lag_hours:
                zone_df[f'{target_col}_lag_{lag}h'] = zone_df[target_col].shift(lag)
                
            # Rolling features
            for window in self.rolling_windows:
                # We shift by 1 before rolling to avoid looking at the current time step (data leakage)
                shifted_target = zone_df[target_col].shift(1)
                zone_df[f'{target_col}_rolling_mean_{window}h'] = shifted_target.rolling(window=window).mean()
                zone_df[f'{target_col}_rolling_std_{window}h'] = shifted_target.rolling(window=window).std()
                
            features_dfs.append(zone_df)
            
        final_df = pd.concat(features_dfs).sort_values(by=['timestamp', 'zone_id']).reset_index(drop=True)
        
        # Drop rows with NaN values created by lag/rolling
        final_df = final_df.dropna()
        
        return final_df

if __name__ == "__main__":
    import os
    if os.path.exists('data/processed/demand.csv'):
        df = pd.read_csv('data/processed/demand.csv', parse_dates=['timestamp'])
        engineer = FeatureEngineer()
        df_featured = engineer.engineer_features(df)
        print(f"Original shape: {df.shape}")
        print(f"Featured shape: {df_featured.shape}")
        df_featured.to_csv('data/processed/demand_featured.csv', index=False)

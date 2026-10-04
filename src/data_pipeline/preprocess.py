import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

class DataPreprocessor:
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scaler = StandardScaler()
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes nulls, sorts chronologically and spatially."""
        df = df.dropna()
        df = df.sort_values(['timestamp', 'zone_id']).reset_index(drop=True)
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronological split to prevent data leakage."""
        timestamps = df['timestamp'].unique()
        timestamps.sort()
        
        n_timestamps = len(timestamps)
        train_end = int(n_timestamps * self.train_ratio)
        val_end = int(n_timestamps * (self.train_ratio + self.val_ratio))
        
        train_times = timestamps[:train_end]
        val_times = timestamps[train_end:val_end]
        test_times = timestamps[val_end:]
        
        train_df = df[df['timestamp'].isin(train_times)].copy()
        val_df = df[df['timestamp'].isin(val_times)].copy()
        test_df = df[df['timestamp'].isin(test_times)].copy()
        
        return train_df, val_df, test_df
    
    def scale(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, columns_to_scale: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fits scaler on train, transforms train, val, test."""
        train_df[columns_to_scale] = self.scaler.fit_transform(train_df[columns_to_scale])
        val_df[columns_to_scale] = self.scaler.transform(val_df[columns_to_scale])
        test_df[columns_to_scale] = self.scaler.transform(test_df[columns_to_scale])
        return train_df, val_df, test_df

if __name__ == "__main__":
    import os
    if os.path.exists('data/processed/demand.csv'):
        df = pd.read_csv('data/processed/demand.csv', parse_dates=['timestamp'])
        processor = DataPreprocessor()
        df_clean = processor.clean(df)
        train, val, test = processor.split_data(df_clean)
        
        cols_to_scale = ['temperature', 'precipitation']
        train, val, test = processor.scale(train, val, test, cols_to_scale)
        
        import joblib
        joblib.dump(processor.scaler, 'data/processed/scaler.pkl')
        
        logging.info(f"Data split shapes - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        
        # Save split data
        train.to_csv('data/processed/train.csv', index=False)
        val.to_csv('data/processed/val.csv', index=False)
        test.to_csv('data/processed/test.csv', index=False)

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

def create_sequences(df: pd.DataFrame, feature_cols: List[str], target_col: str, seq_len: int, pred_len: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences for the ST-GNN.
    Assumes df is sorted by timestamp then zone_id.
    """
    num_zones = df['zone_id'].nunique()
    timestamps = df['timestamp'].unique()
    timestamps.sort()
    
    # Reshape features to [num_timestamps, num_zones, num_features]
    features_arr = np.zeros((len(timestamps), num_zones, len(feature_cols)))
    targets_arr = np.zeros((len(timestamps), num_zones, pred_len))
    
    for i, t in enumerate(timestamps):
        t_df = df[df['timestamp'] == t].sort_values('zone_id')
        features_arr[i, :, :] = t_df[feature_cols].values
        
        # Target is the demand of the next pred_len steps
        # This is a bit tricky if we are at the end, so we handle it below
        
    X, y = [], []
    for i in range(len(timestamps) - seq_len - pred_len + 1):
        # [seq_len, num_zones, num_features] -> [num_zones, seq_len, num_features]
        seq_x = features_arr[i:i+seq_len, :, :]
        seq_x = np.transpose(seq_x, (1, 0, 2))
        
        # [pred_len, num_zones] -> [num_zones, pred_len]
        # the target starts exactly after seq_len
        t_df_target = df[df['timestamp'].isin(timestamps[i+seq_len:i+seq_len+pred_len])].sort_values(['timestamp', 'zone_id'])
        # reshape to [pred_len, num_zones]
        target_vals = t_df_target[target_col].values.reshape((pred_len, num_zones))
        target_vals = np.transpose(target_vals, (1, 0))
        
        X.append(seq_x)
        y.append(target_vals)
        
    return np.array(X), np.array(y)

# We will implement the actual PyTorch training loop in src.evaluation.experimentation or a dedicated run script.
# The `train` function here would handle batching, optimizer, and loss.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def train_st_gnn(model, X_train, y_train, edge_index, edge_weights=None, epochs=20, batch_size=32, lr=0.001):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            
            # Note: batch_X is [batch_size, num_zones, seq_len, num_features]
            # Vectorized batch processing
            bs, num_zones, seq_len, num_features = batch_X.shape
            batch_X_reshaped = batch_X.view(bs * num_zones, seq_len, num_features)
            
            # Duplicate edge_index for the batch to create disjoint subgraphs
            edge_index_batched = torch.cat([edge_index + b * num_zones for b in range(bs)], dim=1)
            
            if edge_weights is not None:
                edge_weights_batched = edge_weights.repeat(bs)
            else:
                edge_weights_batched = None
                
            out = model(batch_X_reshaped, edge_index_batched, edge_weights_batched)
            out = out.view(bs, num_zones, -1)
            
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")
    
    return model

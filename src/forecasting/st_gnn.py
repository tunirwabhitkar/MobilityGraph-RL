import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SpatioTemporalGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for demand forecasting.
    Supports Monte Carlo Dropout for uncertainty estimation.
    """
    def __init__(self, num_node_features: int, hidden_dim: int = 32, num_timesteps_out: int = 1, dropout_rate: float = 0.2):
        super(SpatioTemporalGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Spatial Graph Convolutions
        self.gcn1 = GCNConv(num_node_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Temporal Modeling
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_timesteps_out)

    def forward(self, x, edge_index, edge_weights=None):
        """
        x: [num_nodes, num_timesteps_in, num_features]
        edge_index: [2, num_edges]
        edge_weights: [num_edges] (optional)
        """
        num_nodes, seq_len, num_features = x.shape
        
        # We need to process each timestep through the GCN
        # Alternatively, process everything at once if we reshape
        # Reshape to [num_nodes * seq_len, num_features]
        # But GCN needs graph structure. 
        # Better: loop over sequence length or batch it.
        
        gcn_outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]
            # Spatial convolution
            h = F.relu(self.gcn1(xt, edge_index, edge_weights))
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = F.relu(self.gcn2(h, edge_index, edge_weights))
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            gcn_outputs.append(h)
            
        # Stack temporal outputs: [num_nodes, seq_len, hidden_dim]
        gcn_out_stacked = torch.stack(gcn_outputs, dim=1)
        
        # Temporal processing via LSTM
        lstm_out, (hn, cn) = self.lstm(gcn_out_stacked)
        
        # Take the output of the last timestep
        last_timestep_out = lstm_out[:, -1, :]
        
        # Predict future demand
        out = self.fc(last_timestep_out)
        
        # Predict positive demand
        return F.softplus(out)

    def predict_with_uncertainty(self, x, edge_index, edge_weights=None, n_samples=30):
        """
        Monte Carlo Dropout for uncertainty estimation.
        Requires model to have training=True (handled manually here).
        """
        self.train() # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                preds = self.forward(x, edge_index, edge_weights)
                predictions.append(preds.unsqueeze(0))
                
        predictions = torch.cat(predictions, dim=0) # [n_samples, num_nodes, num_timesteps_out]
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        self.eval() # Revert to eval mode
        
        return mean_pred, lower_bound, upper_bound

import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import haversine_distances
from typing import Tuple, Optional

class GraphBuilder:
    """
    Constructs a spatial graph from geographic zones.
    """
    def __init__(self, method: str = 'knn', k: int = 5, distance_threshold: float = 5.0):
        self.method = method
        self.k = k
        self.distance_threshold = distance_threshold
        
    def build_graph(self, zones_df: pd.DataFrame) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Builds graph based on lat/lon.
        Returns:
            adj_matrix (np.ndarray): Adjacency matrix.
            edge_index (torch.Tensor): PyG edge_index [2, num_edges]
            edge_weights (torch.Tensor): Edge weights based on inverse distance.
        """
        coords = zones_df[['latitude', 'longitude']].values
        num_nodes = len(coords)
        
        # Calculate pairwise Haversine distances in kilometers
        coords_rad = np.radians(coords)
        dist_matrix = haversine_distances(coords_rad, coords_rad) * 6371.0
        
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        if self.method == 'knn':
            # For each node, find k nearest neighbors (excluding itself)
            for i in range(num_nodes):
                # Argsort distances, skip the first one (distance 0 to self)
                nearest_idx = np.argsort(dist_matrix[i])[1:self.k+1]
                adj_matrix[i, nearest_idx] = 1
        elif self.method == 'distance':
            adj_matrix = (dist_matrix < self.distance_threshold).astype(float)
            np.fill_diagonal(adj_matrix, 0)
        else:
            raise ValueError(f"Unknown method {self.method}")
            
        # Create edge index and weights for PyTorch Geometric
        edges = np.argwhere(adj_matrix > 0)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        
        # Weights: inverse distance (normalized)
        weights = []
        for src, dst in edges:
            dist = dist_matrix[src, dst]
            weight = np.exp(-dist) # RBF kernel like weighting
            weights.append(weight)
            
        edge_weights = torch.tensor(weights, dtype=torch.float)
        
        return adj_matrix, edge_index, edge_weights

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    if os.path.exists('data/processed/zones.csv'):
        zones_df = pd.read_csv('data/processed/zones.csv')
        builder = GraphBuilder(method='knn', k=4)
        adj, edge_index, edge_weights = builder.build_graph(zones_df)
        
        logging.info(f"Built graph with {len(zones_df)} nodes and {edge_index.shape[1]} edges.")
        
        # Save graph components
        torch.save({'edge_index': edge_index, 'edge_weights': edge_weights}, 'data/processed/graph.pt')
        np.save('data/processed/adj_matrix.npy', adj)

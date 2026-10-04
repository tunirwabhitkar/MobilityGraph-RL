# import pytest
import numpy as np
import pandas as pd
import torch

from src.anomaly_detection.detector import AnomalyDetector
from src.forecasting.st_gnn import SpatioTemporalGCN
from src.reinforcement_learning.environment import MobilityRebalancingEnv

def test_anomaly_detector_no_leakage():
    detector = AnomalyDetector()
    detector.fit(np.zeros(10)) # dummy fit for isolation forest
    
    # Test data leakage by passing two sets of data where the first half is identical,
    # but the second half is different. The anomaly scores for the first half should be identical.
    expected1 = np.ones(48) * 10
    actual1 = np.ones(48) * 10
    actual1[24] = 100 # anomaly at t=24
    
    df1 = detector.detect(expected1, actual1)
    
    expected2 = np.ones(48) * 10
    actual2 = np.ones(48) * 10
    actual2[24] = 100
    actual2[40:] = 500 # wild changes in the future
    
    df2 = detector.detect(expected2, actual2)
    
    # The anomaly score for the first 25 points should be exactly the same
    np.testing.assert_array_almost_equal(df1['z_score'].iloc[:25].values, df2['z_score'].iloc[:25].values)

def test_st_gnn_shapes():
    num_nodes = 25
    seq_len = 12
    num_features = 5
    hidden_dim = 32
    
    model = SpatioTemporalGCN(num_node_features=num_features, hidden_dim=hidden_dim)
    
    # Dummy input
    x = torch.randn(num_nodes, seq_len, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 50))
    
    out = model(x, edge_index)
    assert out.shape == (num_nodes, 1), f"Expected shape {(num_nodes, 1)}, got {out.shape}"
    
    # Test MC dropout predict
    mean_pred, lb, ub = model.predict_with_uncertainty(x, edge_index, n_samples=5)
    assert mean_pred.shape == (num_nodes, 1)
    assert lb.shape == (num_nodes, 1)
    assert ub.shape == (num_nodes, 1)
    # lb should be <= mean_pred <= ub
    assert torch.all(lb <= mean_pred + 1e-5)
    assert torch.all(mean_pred <= ub + 1e-5)

def test_rl_environment_capacities():
    num_zones = 3
    num_steps = 10
    
    demand = np.zeros((num_steps, num_zones))
    supply = np.ones((num_steps, num_zones)) * 50
    dist_matrix = np.ones((num_zones, num_zones))
    np.fill_diagonal(dist_matrix, 0)
    
    capacities = np.array([60, 100, 100]) # zone 0 can only take 10 more
    
    env = MobilityRebalancingEnv(demand, supply, dist_matrix, zone_capacities=capacities, max_relocations_per_step=20)
    env.reset()
    
    # action = +1 (pull max relocations) for all zones
    # But zone 0 only has space for 10. Max relocations is 20.
    # We should see pull constrained.
    # Wait, if all pull, no one pushes. So transfer volume = 0.
    # We need someone to push.
    action = np.array([1.0, 1.0, -1.0]) # zone 0 and 1 pull, zone 2 pushes
    # Zone 2 pushes 20.
    # Zone 0 wants 20, Zone 1 wants 20.
    # Zone 0 is capped at available space = 10.
    # So push = 20, pull = [10, 20, 0] -> total pull = 30.
    # transfer_volume = min(20, 30) = 20.
    # Pull gets scaled: Zone 0 gets 10 * 20/30 = 6.66, Zone 1 gets 20 * 20/30 = 13.33.
    
    _, _, _, _, _ = env.step(action)
    
    assert env.current_supply[0] <= 60.01, f"Zone 0 exceeded capacity: {env.current_supply[0]}"

if __name__ == "__main__":
    test_anomaly_detector_no_leakage()
    test_st_gnn_shapes()
    test_rl_environment_capacities()
    print("All tests passed!")

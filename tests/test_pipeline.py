import pytest
import numpy as np
import pandas as pd

from src.data_pipeline.simulator import MobilityDataSimulator
from src.optimization.imbalance import ImbalanceEngine
from src.optimization.rebalancing import RebalancingOptimizer

def test_simulator_deterministic():
    sim1 = MobilityDataSimulator(num_zones=5, num_days=1, seed=42)
    zones1, demand1 = sim1.run()
    
    sim2 = MobilityDataSimulator(num_zones=5, num_days=1, seed=42)
    zones2, demand2 = sim2.run()
    
    pd.testing.assert_frame_equal(zones1, zones2)
    pd.testing.assert_frame_equal(demand1, demand2)

def test_imbalance_engine():
    engine = ImbalanceEngine()
    
    # 3 zones
    predictions = np.array([100, 50, 10])
    supply = np.array([80, 50, 20])
    zone_ids = np.array([0, 1, 2])
    
    df = engine.calculate_gap(predictions, supply, zone_ids)
    
    assert df.loc[0, 'demand_gap'] == 20
    assert df.loc[0, 'status'] == 'CRITICAL SHORTAGE'
    
    assert df.loc[1, 'demand_gap'] == 0
    assert df.loc[1, 'status'] == 'BALANCED'
    
    assert df.loc[2, 'demand_gap'] == -10
    assert df.loc[2, 'status'] == 'SURPLUS'

def test_rebalancing_optimizer():
    optimizer = RebalancingOptimizer(cost_per_km=1.0)
    
    demand = np.array([100, 10])
    supply = np.array([50, 60])
    dist_matrix = np.array([[0, 5], [5, 0]])
    
    # Zone 0 needs 50. Zone 1 has 50 extra.
    # It should move 50 from Zone 1 to Zone 0.
    relocations, metrics = optimizer.optimize(demand, supply, dist_matrix)
    
    assert relocations[1, 0] == 50
    assert relocations[0, 1] == 0
    assert metrics['total_unmet_demand'] == 0

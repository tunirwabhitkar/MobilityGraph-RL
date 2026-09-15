from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
from typing import Tuple

class RebalancingOptimizer:
    """
    Optimizes vehicle relocations to minimize cost and unmet demand.
    """
    def __init__(self, cost_per_km: float = 0.5, unmet_penalty: float = 10.0, idle_penalty: float = 1.0):
        self.cost_per_km = cost_per_km
        self.unmet_penalty = unmet_penalty
        self.idle_penalty = idle_penalty
        
    def optimize(self, 
                 demand: np.ndarray, 
                 supply: np.ndarray, 
                 dist_matrix: np.ndarray, 
                 max_relocation_dist: float = 10.0) -> Tuple[np.ndarray, dict]:
        """
        demand: [num_zones] Expected demand
        supply: [num_zones] Available vehicles
        dist_matrix: [num_zones, num_zones] Distance between zones
        """
        num_zones = len(demand)
        solver = pywraplp.Solver.CreateSolver('SCIP')
        
        if not solver:
            return None, {}
            
        # Decision Variables
        # x[i][j]: number of vehicles moved from i to j
        x = {}
        for i in range(num_zones):
            for j in range(num_zones):
                if dist_matrix[i, j] <= max_relocation_dist:
                    x[i, j] = solver.IntVar(0, solver.infinity(), f'x_{i}_{j}')
                else:
                    # Prevent long distance relocations
                    x[i, j] = solver.IntVar(0, 0, f'x_{i}_{j}')
                    
        # Unmet demand in zone j
        unmet = {}
        for j in range(num_zones):
            unmet[j] = solver.NumVar(0, solver.infinity(), f'unmet_{j}')
            
        # Idle vehicles in zone j
        idle = {}
        for j in range(num_zones):
            idle[j] = solver.NumVar(0, solver.infinity(), f'idle_{j}')
            
        # Constraints
        # 1. Cannot move more vehicles out of i than are available
        for i in range(num_zones):
            solver.Add(sum(x[i, j] for j in range(num_zones)) <= supply[i])
            
        # 2. Final supply at j = Initial supply - moved out + moved in
        # Unmet demand >= predicted_demand - final_supply
        # Idle vehicles >= final_supply - predicted_demand
        for j in range(num_zones):
            final_supply = supply[j] - sum(x[j, k] for k in range(num_zones)) + sum(x[i, j] for i in range(num_zones))
            solver.Add(unmet[j] >= demand[j] - final_supply)
            solver.Add(idle[j] >= final_supply - demand[j])
            
        # Objective
        objective = solver.Objective()
        for i in range(num_zones):
            for j in range(num_zones):
                if i != j:
                    objective.SetCoefficient(x[i, j], dist_matrix[i, j] * self.cost_per_km)
            
            objective.SetCoefficient(unmet[i], self.unmet_penalty)
            objective.SetCoefficient(idle[i], self.idle_penalty)
            
        objective.SetMinimization()
        
        status = solver.Solve()
        
        relocations = np.zeros((num_zones, num_zones))
        metrics = {}
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for i in range(num_zones):
                for j in range(num_zones):
                    relocations[i, j] = x[i, j].solution_value()
            
            metrics = {
                'total_cost': solver.Objective().Value(),
                'total_unmet_demand': sum(unmet[j].solution_value() for j in range(num_zones)),
                'total_relocations': np.sum(relocations)
            }
        
        return relocations, metrics

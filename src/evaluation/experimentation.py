import pandas as pd
import numpy as np

class ExperimentRunner:
    """Runs and compares different rebalancing strategies."""
    
    def __init__(self, demand: np.ndarray, initial_supply: np.ndarray, dist_matrix: np.ndarray):
        self.demand = demand
        self.initial_supply = initial_supply
        self.dist_matrix = dist_matrix
        self.num_steps = len(demand)
        
    def evaluate_strategy(self, strategy_func, name: str) -> dict:
        """
        Evaluates a given strategy function over the horizon.
        strategy_func takes (current_demand, current_supply) and returns (relocations, next_supply)
        """
        current_supply = np.copy(self.initial_supply[0])
        total_unmet = 0
        total_relocations = 0
        total_cost = 0
        
        for t in range(self.num_steps):
            current_demand = self.demand[t]
            
            # Agent decides relocations based on current state (and maybe predictions in a real system)
            relocations = strategy_func(current_demand, current_supply, t)
            
            # Apply relocations
            moved_out = relocations.sum(axis=1)
            moved_in = relocations.sum(axis=0)
            current_supply = current_supply - moved_out + moved_in
            
            # Evaluate step
            unmet = np.maximum(0, current_demand - current_supply)
            total_unmet += np.sum(unmet)
            
            cost = np.sum(relocations * self.dist_matrix) * 0.5 # $0.5 per km
            total_cost += cost
            total_relocations += np.sum(relocations)
            
            # Transition to next natural state
            # (In simulation, supply is updated based on natural movement)
            if t < self.num_steps - 1:
                natural_delta = self.initial_supply[t+1] - self.initial_supply[t]
                current_supply = np.maximum(0, current_supply + natural_delta - current_demand)
                
        return {
            'Strategy': name,
            'Total Unmet Demand': total_unmet,
            'Total Relocations': total_relocations,
            'Total Cost ($)': total_cost
        }
        
    def run_comparison(self, strategies: dict) -> pd.DataFrame:
        """Runs all strategies and returns a comparison DataFrame."""
        results = []
        for name, func in strategies.items():
            res = self.evaluate_strategy(func, name)
            results.append(res)
            
        return pd.DataFrame(results)

# Example strategies for the runner
def no_rebalancing_strategy(demand, supply, t):
    num_zones = len(demand)
    return np.zeros((num_zones, num_zones))

def greedy_rebalancing_strategy(demand, supply, t, max_dist_idx):
    """Simple greedy: move from surplus to nearest shortage."""
    num_zones = len(demand)
    relocations = np.zeros((num_zones, num_zones))
    
    gap = demand - supply
    surplus_zones = np.where(gap < -5)[0]
    shortage_zones = np.where(gap > 5)[0]
    
    temp_supply = np.copy(supply)
    
    for s_zone in surplus_zones:
        available_to_move = -gap[s_zone] - 5
        if available_to_move <= 0:
            continue
            
        # Find nearest shortage
        nearest_targets = max_dist_idx[s_zone]
        
        for t_zone in nearest_targets:
            if t_zone in shortage_zones:
                needed = gap[t_zone] - 5
                if needed > 0:
                    moved = min(available_to_move, needed)
                    relocations[s_zone, t_zone] += moved
                    available_to_move -= moved
                    gap[t_zone] -= moved
                    
                if available_to_move <= 0:
                    break
                    
    return relocations

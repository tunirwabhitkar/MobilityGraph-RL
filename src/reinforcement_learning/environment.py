import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MobilityRebalancingEnv(gym.Env):
    """
    Custom Environment for dynamic fleet rebalancing.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, demand_data: np.ndarray, supply_data: np.ndarray, dist_matrix: np.ndarray, 
                 zone_capacities: np.ndarray = None,
                 max_relocations_per_step: int = 50, cost_per_km: float = 0.5, unmet_penalty: float = 10.0):
        super().__init__()
        
        self.demand_data = demand_data  # [num_steps, num_zones]
        self.supply_data = supply_data  # [num_steps, num_zones] (initial scenario supply)
        self.dist_matrix = dist_matrix
        
        self.num_steps, self.num_zones = demand_data.shape
        self.current_step = 0
        
        if zone_capacities is None:
            self.zone_capacities = np.ones(self.num_zones) * 100
        else:
            self.zone_capacities = zone_capacities
            
        self.max_relocations = max_relocations_per_step
        self.cost_per_km = cost_per_km
        self.unmet_penalty = unmet_penalty
        
        # Action space: desired change in supply per zone (normalized)
        # shape: (num_zones,)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_zones,), dtype=np.float32)
        
        # Observation space: 
        # [demand_t, supply_t, time_of_day]
        obs_dim = self.num_zones * 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_supply = np.copy(self.supply_data[0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_supply = np.copy(self.supply_data[self.current_step])
        
        return self._get_obs(), {}

    def _get_obs(self):
        demand = self.demand_data[self.current_step]
        time_of_day = (self.current_step % 24) / 24.0
        return np.concatenate([demand, self.current_supply, [time_of_day]]).astype(np.float32)

    def step(self, action):
        # Action is [num_zones] in [-1.0, 1.0]
        delta = action * self.max_relocations
        
        push = np.where(delta < 0, -delta, 0)
        pull = np.where(delta > 0, delta, 0)
        
        push = np.minimum(push, self.current_supply)
        available_space = np.maximum(0, self.zone_capacities - self.current_supply)
        pull = np.minimum(pull, available_space)
        
        total_push = np.sum(push)
        total_pull = np.sum(pull)
        
        transfer_volume = min(total_push, total_pull)
        
        if transfer_volume > 0:
            push = push * (transfer_volume / total_push)
            pull = pull * (transfer_volume / total_pull)
            self.current_supply = self.current_supply - push + pull
            
            # Approximate relocation cost
            avg_dist = np.mean(self.dist_matrix)
            relocation_cost = transfer_volume * avg_dist * self.cost_per_km
            total_relocations = transfer_volume
        else:
            relocation_cost = 0.0
            total_relocations = 0.0
        
        # Calculate Reward
        current_demand = self.demand_data[self.current_step]
        
        unmet_demand = np.maximum(0, current_demand - self.current_supply)
        total_unmet = np.sum(unmet_demand)
        
        
        reward = -(total_unmet * self.unmet_penalty + relocation_cost)
        
        # Natural transition to next step
        # In reality, supply also changes due to customer trips.
        # We approximate by transitioning to the simulator's next supply state, 
        # plus any benefits/deficits we caused compared to baseline.
        baseline_unmet = np.maximum(0, current_demand - self.supply_data[self.current_step])
        saved_unmet = np.sum(baseline_unmet) - total_unmet
        
        self.current_step += 1
        terminated = self.current_step >= self.num_steps - 1
        truncated = False
        
        if not terminated:
            # We mix the true simulated state with our interventions
            # For simplicity in this env, we just carry over our supply + natural delta
            natural_delta = self.supply_data[self.current_step] - self.supply_data[self.current_step - 1]
            self.current_supply = np.maximum(0, self.current_supply + natural_delta - current_demand)
            
        info = {
            'unmet_demand': total_unmet,
            'relocation_cost': relocation_cost,
            'saved_unmet': saved_unmet,
            'total_relocations': total_relocations
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

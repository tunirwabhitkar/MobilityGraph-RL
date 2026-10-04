import logging
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.reinforcement_learning.environment import MobilityRebalancingEnv
import os

def train_ppo(demand_data: np.ndarray, supply_data: np.ndarray, dist_matrix: np.ndarray, 
              total_timesteps: int = 10000, model_save_path: str = 'models/ppo_rebalancing'):
    """
    Trains a PPO agent for the fleet rebalancing environment.
    """
    
    # Create the environment
    def make_env():
        return MobilityRebalancingEnv(
            demand_data=demand_data, 
            supply_data=supply_data, 
            dist_matrix=dist_matrix
        )
        
    env = DummyVecEnv([make_env])
    
    # Initialize PPO
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64)
    
    logging.info(f"Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    logging.info(f"Model saved to {model_save_path}.zip")
    
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage if data exists
    if os.path.exists('data/processed/demand.csv') and os.path.exists('data/processed/adj_matrix.npy'):
        demand_df = pd.read_csv('data/processed/demand.csv')
        dist_matrix = np.load('data/processed/adj_matrix.npy') # Approximate for now, assume adj matrix represents distance
        
        # Reshape demand into [num_steps, num_zones]
        num_zones = demand_df['zone_id'].nunique()
        steps = len(demand_df) // num_zones
        
        # Just use subset for testing
        demand_data = demand_df['demand'].values[:steps*num_zones].reshape((steps, num_zones))
        supply_data = demand_df['available_vehicles'].values[:steps*num_zones].reshape((steps, num_zones))
        
        train_ppo(demand_data, supply_data, dist_matrix, total_timesteps=5000)

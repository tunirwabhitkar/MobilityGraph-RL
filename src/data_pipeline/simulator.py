import numpy as np
import pandas as pd
from typing import Tuple

class MobilityDataSimulator:
    """
    Deterministically simulates a mobility network, demand, fleet states, and external features.
    """
    def __init__(self, num_zones: int = 25, num_days: int = 30, seed: int = 42):
        self.num_zones = num_zones
        self.num_days = num_days
        self.num_hours = num_days * 24
        self.seed = seed
        np.random.seed(self.seed)
        
    def generate_zones(self) -> pd.DataFrame:
        """Generates random coordinates for zones within a city-like grid."""
        # Assume a city area of roughly 10x10 km
        lats = np.random.uniform(40.70, 40.80, self.num_zones)
        lons = np.random.uniform(-74.05, -73.95, self.num_zones)
        
        # Base demand characteristics per zone
        # Some zones are high demand (commercial), some low (residential)
        zone_types = np.random.choice(['residential', 'commercial', 'mixed'], self.num_zones, p=[0.5, 0.2, 0.3])
        base_demand_multiplier = np.where(zone_types == 'commercial', 2.5, 
                                          np.where(zone_types == 'residential', 1.0, 1.5))
        
        zones_df = pd.DataFrame({
            'zone_id': range(self.num_zones),
            'latitude': lats,
            'longitude': lons,
            'zone_type': zone_types,
            'base_demand': base_demand_multiplier
        })
        return zones_df

    def generate_temporal_features(self) -> pd.DataFrame:
        """Generates datetime features and weather."""
        dates = pd.date_range(start='2023-01-01', periods=self.num_hours, freq='H')
        df = pd.DataFrame({'timestamp': dates})
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Simulate weather: temp varies daily and seasonally
        base_temp = 15.0 + 10.0 * np.sin(2 * np.pi * df['timestamp'].dt.dayofyear / 365.0)
        daily_variation = 5.0 * np.sin(2 * np.pi * (df['hour'] - 6) / 24.0)
        df['temperature'] = base_temp + daily_variation + np.random.normal(0, 2, self.num_hours)
        
        # Precipitation probability
        df['precipitation'] = np.where(np.random.random(self.num_hours) < 0.1, np.random.exponential(5, self.num_hours), 0)
        
        return df

    def generate_demand(self, zones_df: pd.DataFrame, time_df: pd.DataFrame) -> pd.DataFrame:
        """Generates zone-level demand incorporating temporal patterns and weather effects."""
        records = []
        for _, zone in zones_df.iterrows():
            zone_id = zone['zone_id']
            base = zone['base_demand']
            z_type = zone['zone_type']
            
            for _, t in time_df.iterrows():
                hour = t['hour']
                is_weekend = t['is_weekend']
                precip = t['precipitation']
                
                # Diurnal pattern based on zone type
                if z_type == 'residential':
                    # Morning peak out, evening peak in (but demand represents pickups here)
                    hour_factor = 1.0 + 1.5 * np.exp(-0.5 * ((hour - 8)/1.5)**2) + 0.8 * np.exp(-0.5 * ((hour - 18)/2.0)**2)
                elif z_type == 'commercial':
                    # Evening peak out
                    hour_factor = 0.5 + 2.5 * np.exp(-0.5 * ((hour - 17)/2.0)**2) + 1.0 * np.exp(-0.5 * ((hour - 12)/2.0)**2)
                else: # mixed
                    hour_factor = 1.0 + 1.0 * np.exp(-0.5 * ((hour - 9)/2.0)**2) + 1.0 * np.exp(-0.5 * ((hour - 18)/2.0)**2)
                
                # Weekend effect
                weekend_factor = 0.6 if is_weekend else 1.0
                
                # Weather effect: rain reduces demand
                weather_factor = max(0.5, 1.0 - 0.05 * precip)
                
                # Combine effects with some noise
                mean_demand = 10 * base * hour_factor * weekend_factor * weather_factor
                actual_demand = max(0, int(np.random.poisson(mean_demand)))
                
                # Fleet state simulation
                available_vehicles = max(0, int(actual_demand + np.random.normal(5, 10))) # Some gap naturally occurs
                
                records.append({
                    'timestamp': t['timestamp'],
                    'zone_id': zone_id,
                    'demand': actual_demand,
                    'available_vehicles': available_vehicles,
                    'temperature': t['temperature'],
                    'precipitation': t['precipitation']
                })
                
        return pd.DataFrame(records)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Generating zones...")
        zones = self.generate_zones()
        print("Generating temporal features...")
        time_features = self.generate_temporal_features()
        print("Simulating demand and fleet states...")
        demand = self.generate_demand(zones, time_features)
        return zones, demand

if __name__ == "__main__":
    simulator = MobilityDataSimulator()
    zones, demand = simulator.run()
    
    # Save to standard data directory
    import os
    os.makedirs('data/processed', exist_ok=True)
    zones.to_csv('data/processed/zones.csv', index=False)
    demand.to_csv('data/processed/demand.csv', index=False)
    print("Simulation complete. Data saved to data/processed/")

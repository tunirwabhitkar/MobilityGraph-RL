import folium
import pandas as pd
from folium.plugins import HeatMap

class GeoMapper:
    """Generates geographical visualizations for demand and routes."""
    
    def __init__(self, center_lat: float = 40.75, center_lon: float = -74.00, zoom_start: int = 12):
        self.center = [center_lat, center_lon]
        self.zoom_start = zoom_start
        
    def plot_heatmap(self, zones_df: pd.DataFrame, value_col: str = 'demand', radius: int = 15) -> folium.Map:
        """Plots a heatmap of the specified value column."""
        m = folium.Map(location=self.center, zoom_start=self.zoom_start)
        
        # Prepare data for HeatMap: [lat, lon, weight]
        heat_data = [[row['latitude'], row['longitude'], row[value_col]] 
                     for index, row in zones_df.iterrows() if row[value_col] > 0]
                     
        if heat_data:
            HeatMap(heat_data, radius=radius).add_to(m)
            
        return m
        
    def plot_routes(self, zones_df: pd.DataFrame, routes: list) -> folium.Map:
        """Plots the relocation routes."""
        m = folium.Map(location=self.center, zoom_start=self.zoom_start)
        
        # Add markers for all zones
        for _, row in zones_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color='blue',
                fill=True,
                popup=f"Zone {row['zone_id']}"
            ).add_to(m)
            
        colors = ['red', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']
        
        for i, route in enumerate(routes):
            color = colors[i % len(colors)]
            route_coords = []
            for node_id in route:
                node_data = zones_df[zones_df['zone_id'] == node_id].iloc[0]
                route_coords.append([node_data['latitude'], node_data['longitude']])
                
            # Draw line
            folium.PolyLine(
                route_coords,
                color=color,
                weight=3,
                opacity=0.8,
                tooltip=f"Vehicle {i+1} Route"
            ).add_to(m)
            
        return m

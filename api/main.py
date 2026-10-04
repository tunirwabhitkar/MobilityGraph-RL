from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import os
from src.forecasting.st_gnn import SpatioTemporalGCN

app = FastAPI(title="MobilityGraph-RL API", version="1.0")

NUM_ZONES = 25
NUM_FEATURES = 5 # Example number of features

model = SpatioTemporalGCN(num_node_features=NUM_FEATURES)
MODEL_PATH = "data/models/st_gnn.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Dummy static graph for inference (in a real app, this is built from zones)
dummy_edge_index = torch.randint(0, NUM_ZONES, (2, 50))

class ForecastRequest(BaseModel):
    zone_id: str
    forecast_horizon: str
    
class ForecastResponse(BaseModel):
    zone_id: str
    forecast_horizon: str
    predicted_demand: float
    lower_bound: float
    upper_bound: float

class OptimizeRequest(BaseModel):
    demand: list
    supply: list
    
class OptimizeResponse(BaseModel):
    relocations: list
    total_cost: float
    unmet_demand: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/forecast", response_model=ForecastResponse)
def get_forecast(req: ForecastRequest):
    # In a real app, we would query the feature store for the last `seq_len` timesteps for all zones
    dummy_x = torch.randn(NUM_ZONES, 24, NUM_FEATURES)
    
    with torch.no_grad():
        mean_pred, lower_bound, upper_bound = model.predict_with_uncertainty(dummy_x, dummy_edge_index, n_samples=10)
        
    try:
        zone_idx = int(req.zone_id)
    except ValueError:
        zone_idx = 0
    zone_idx = min(max(0, zone_idx), NUM_ZONES - 1)
    
    predicted = float(mean_pred[zone_idx, 0].item())
    lb = float(lower_bound[zone_idx, 0].item())
    ub = float(upper_bound[zone_idx, 0].item())

    return ForecastResponse(
        zone_id=req.zone_id,
        forecast_horizon=req.forecast_horizon,
        predicted_demand=max(0.0, predicted),
        lower_bound=max(0.0, lb),
        upper_bound=max(0.0, ub)
    )

@app.post("/optimize-rebalancing", response_model=OptimizeResponse)
def optimize_rebalancing(req: OptimizeRequest):
    from src.optimization.rebalancing import RebalancingOptimizer
    # Mock distance matrix for demo
    num_zones = len(req.demand)
    dist_matrix = np.ones((num_zones, num_zones))
    np.fill_diagonal(dist_matrix, 0)
    
    optimizer = RebalancingOptimizer()
    relocations, metrics = optimizer.optimize(
        demand=np.array(req.demand),
        supply=np.array(req.supply),
        dist_matrix=dist_matrix
    )
    
    if relocations is None:
        raise HTTPException(status_code=500, detail="Optimization failed.")
        
    return OptimizeResponse(
        relocations=relocations.tolist(),
        total_cost=metrics.get('total_cost', 0),
        unmet_demand=metrics.get('total_unmet_demand', 0)
    )

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="MobilityGraph-RL API", version="1.0")

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
    # In a real app, this would query the loaded ST-GNN model
    # Mock response for architecture completeness
    predicted = float(np.random.poisson(100))
    return ForecastResponse(
        zone_id=req.zone_id,
        forecast_horizon=req.forecast_horizon,
        predicted_demand=predicted,
        lower_bound=max(0, predicted - 20),
        upper_bound=predicted + 20
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

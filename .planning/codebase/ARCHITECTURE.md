---
last_mapped_commit: 855061963e1939f23d879d2f08bdac2fb130828b
last_mapped_at: 2026-10-05
---
# ARCHITECTURE.md

**Analysis Date:** 2026-10-05

## System Architecture

MobilityGraph-RL is a modular pipeline combining ST-GNNs (Spatio-Temporal Graph Neural Networks), MILP (Mixed Integer Linear Programming) optimization, and RL (Reinforcement Learning) for mobility and vehicle routing.

### Key Layers & Components

1. **Data Pipeline**: Simulator (`simulator.py`) generates demand; Preprocessor (`preprocess.py`) prepares data.
2. **Feature Engineering**: Feature extraction (`features.py`) adds lags and metrics.
3. **Graph ML**: Graph builder (`graph_builder.py`) constructs spatial relationships; Anomaly detector (`detector.py`).
4. **Forecasting**: Baselines (`baselines.py`), training loop (`train.py`), and the core ST-GNN model (`st_gnn.py`).
5. **Optimization**: Rebalancing MILP (`rebalancing.py`) and experiments (`experimentation.py`).
6. **Reinforcement Learning**: Environment (`environment.py`) and PPO training loop (`train_rl.py`).
7. **APIs & Dashboards**: FastAPI backend (`api/main.py`) and Streamlit frontend (`dashboard/app.py`).

## Data Flow

- `MobilityDataSimulator` -> `DataPreprocessor` -> `FeatureEngineer`
- `GraphBuilder` consumes preprocessed data to create edge indices for the ST-GNN.
- `ST-GNN` makes traffic predictions.
- These predictions feed into either the `MILP Optimizer` or the `RL Environment` for vehicle rebalancing.

<!-- refreshed: 2026-10-05 -->

---
last_mapped_commit: 855061963e1939f23d879d2f08bdac2fb130828b
last_mapped_at: 2026-10-05
---
# STRUCTURE.md

**Analysis Date:** 2026-10-05

## Directory Layout

- `api/` - FastAPI endpoints (e.g., `main.py`).
- `configs/` - Project configuration files (currently mostly empty).
- `dashboard/` - Streamlit app (`app.py`).
- `data/` - Raw, processed, and model artifacts.
- `notebooks/` - Jupyter notebooks for exploration.
- `src/` - Core source code divided into:
  - `data_pipeline/`: Simulator and preprocessing.
  - `feature_engineering/`: Feature extraction.
  - `forecasting/`: Model architecture (`st_gnn.py`) and training (`train.py`, `baselines.py`).
  - `graph_ml/`: Graph building and anomaly detection.
  - `monitoring/`: Data drift tracking.
  - `optimization/`: MILP and experimentation.
  - `reinforcement_learning/`: Custom Gym environment and PPO training.
- `tests/` - Pytest test suite (e.g., `test_pipeline.py`).

<!-- refreshed: 2026-10-05 -->

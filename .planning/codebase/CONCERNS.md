---
last_mapped_commit: 855061963e1939f23d879d2f08bdac2fb130828b
last_mapped_at: 2026-10-05
---
# CONCERNS.md

**Analysis Date:** 2026-10-05

## Technical Debt & Issues

The codebase has significant data science bugs and ML anti-patterns:
- **`traffic_prediction.py`**: A fundamentally flawed legacy script with target mismatch and data copying bugs. Needs deletion or archiving.
- **Global Random State**: `simulator.py` sets a global `np.random.seed(42)`, polluting the random state for downstream modules (affects tests and training).
- **RL Action Space**: The continuous action space in `environment.py` is too large (625-D) for PPO to converge effectively.
- **Data Leakage**: `preprocess.py` scales data via `SettingWithCopyWarning` and loses the scaler. `detector.py` leaks test statistics during Z-score refitting. `baselines.py` needs sorting assertions to prevent future data leakage.
- **Performance**: `simulator.py` uses O(n^2) nested loops for data generation, preventing scalability. `train.py` iterates over batch elements sequentially, nullifying GPU vectorization.
- **Graph Inaccuracy**: `graph_builder.py` uses uncorrected Euclidean distance for lat/lon, causing geographical distortion.
- **Missing Configs & Versions**: No `requirements.txt` version pinning and hardcoded hyperparameters everywhere.

<!-- refreshed: 2026-10-05 -->

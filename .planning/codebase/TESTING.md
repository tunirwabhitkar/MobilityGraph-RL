---
last_mapped_commit: 855061963e1939f23d879d2f08bdac2fb130828b
last_mapped_at: 2026-10-05
---
# TESTING.md

**Analysis Date:** 2026-10-05

## Testing Strategy

- **Framework**: Pytest
- **Location**: `tests/` directory (e.g., `test_pipeline.py`).

## Known Issues

- Tests exhibit deterministic fragility (global `np.random.seed` dependencies).
- Coverage appears incomplete (missing dedicated tests for ST-GNN and graph building).

<!-- refreshed: 2026-10-05 -->

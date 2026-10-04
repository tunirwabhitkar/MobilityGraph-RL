---
last_mapped_commit: 855061963e1939f23d879d2f08bdac2fb130828b
last_mapped_at: 2026-10-05
---
# CONVENTIONS.md

**Analysis Date:** 2026-10-05

## Coding Standards

- **Style**: Pythonic, but currently lacking strict linters or formatters.
- **Error Handling**: Missing formal logging, heavily relies on `print()`.
- **Typing**: Some type hints are present but not uniformly enforced.

## Notable Patterns

- The codebase uses standard object-oriented patterns for the data processing pipeline (classes with `run()`, `fit_transform()`, etc.).
- The RL component uses OpenAI `gymnasium` (formerly `gym`) conventions.
- Deep learning is written in PyTorch + PyTorch Geometric (`torch_geometric`).

<!-- refreshed: 2026-10-05 -->

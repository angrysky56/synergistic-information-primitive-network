# Phase 2: Mathematical Stability & Estimator Refinement

## Goal
Ensure Matrix-Rényi estimators are robust, numerically stable, and computationally efficient.

## Objectives
1. **Audit `ais_estimator.py` and `pid_estimator.py`**: Check for small-eigenvalue sensitivity and numerical instability in Matrix-Rényi entropy calculations.
2. **Implement/Verify $O(N^2)$ approximations**: Evaluate and potentially implement Randomized Trace or Polynomial approximations to improve performance over $O(N^3)$ eigenvalue decomposition.
3. **Boundary Condition Testing**: Add tests for fractional alpha-entropy calculations, focusing on edge cases (e.g., zero variance, highly correlated inputs).

## Current Implementations
- `src/sipnet/infrastructure/information_theory/ais_estimator.py`
- `src/sipnet/infrastructure/information_theory/pid_estimator.py`
- `src/sipnet/infrastructure/information_theory/kernel_utils.py`

## Constraints
- Must maintain compatibility with `torch`.
- Must satisfy strict typing requirements established in Phase 1.
- Performance is a secondary goal to stability in this phase, but $O(N^2)$ approximations should be explored.

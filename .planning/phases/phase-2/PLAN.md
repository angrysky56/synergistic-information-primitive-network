# Phase 2 Plan: Mathematical Stability & Estimator Refinement

## Context
Phase 2 focuses on hardening the information-theoretic estimators. Currently, the system uses Matrix-Rényi entropy with $\alpha=2$, which is $O(N^2)$ but limited. We need to support fractional $\alpha$ and ensure numerical stability across edge cases.

## Tasks

### 2.1 Audit & Stabilization (Plan 2.1)
- [x] **Stabilize Kernel Normalization**: Ensure `compute_normalized_kernel` is robust to zero-variance batches.
- [x] **Add Epsilon to Eigenvalues**: When using eigenvalue-based estimators, ensure a small epsilon is added to avoid $\log(0)$.
- [x] **Test for Singularities**: Create unit tests for degenerate cases (all zeros, all identical, single sample).

### 2.2 Generalized Matrix-Rényi Estimator (Plan 2.2)
- [x] **Implement Exact Alpha-Entropy**: Create a function that calculates $H_\alpha$ using eigenvalue decomposition ($O(N^3)$) as a reference.
- [x] **Implement $O(N^2)$ Approximation**:
    - [x] Use Chebyshev polynomial expansion to approximate $Tr(A^\alpha)$.
    - [x] Alternatively, implement a Randomized Trace estimator for $H_\alpha$.
- [x] **Implement von Neumann Entropy**: Add $H_1(A) = -Tr(A \log_2 A)$ using a stable approximation.

### 2.3 Verification & Boundary Tests (Plan 2.3)
- [x] **Cross-Validate Approximations**: Compare $O(N^2)$ results against $O(N^3)$ exact results.
- [x] **Gradient Flow Check**: Verify that backpropagation through the new estimators is stable and does not produce NaNs.
- [x] **Batch Size Scaling**: Verify that estimators remain stable as batch size $N$ varies from 2 to 1024.

## Success Criteria
- No NaNs produced by any estimator on edge-case data.
- $O(N^2)$ approximations are within 1% error of $O(N^3)$ exact calculations.
- All Phase 2 requirements in `ROADMAP.md` and `REQUIREMENTS.md` are satisfied.

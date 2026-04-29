# Technical Concerns & Risks

## Current Risks
- **Spectral Radius Stability**: Recurrent weights in `StorageNode` rely on spectral radius normalization. If the linear algebra solver fails or produces unstable eigenvalues, the node state may explode or vanish.
- **Computational Complexity**: The information-theoretic estimators (PID, TE, AIS) are integrated into the training loop. These calculations can be significantly slower than standard backpropagation, potentially bottlenecking training.
- **Bleeding Edge Dependencies**: Using Python 3.13 and PyTorch 2.10.0 requires careful environment management as these versions are very recent or future-dated.

## Technical Debt
- **Interface Placeholders**: `src/sipnet/interfaces/api` is currently empty. The CLI is in its early stages.
- **Documentation Gaps**: While docstrings exist, the high-level math behind the "Synergistic Information Primitive" needs better documentation for onboarding.
- **Test Coverage**: Unit tests are present for primitive nodes, but integration tests for deep hierarchies (`num_layers > 5`) and top-down routing are missing.

## Future Challenges
- **Scalability**: Scaling the number of storage nodes and parallel buses may lead to memory issues during the BPTT (Backpropagation Through Time) phase.
- **Catastrophic Forgetting**: As discussed in research notes, the synergy-hub dynamics need to be hardened against forgetting previously learned synergistic patterns.

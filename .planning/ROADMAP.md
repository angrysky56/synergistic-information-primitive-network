# SIP-Net: Development Roadmap

## Milestone 1: Architecture Hardening (Current)
Focus on stability, validity, and clean foundations.

### Phase 1: Diagnostic Audit & Typing Hardening
Achieve strict typing and resolve structural bug noise.
- [ ] **Plan 1.1**: Run full `mypy` and `pytest` suite to categorize all current failures.
- [ ] **Plan 1.2**: Resolve `torch.nn.Module` typing conflicts in `domain/nodes`.
- [ ] **Plan 1.3**: Fix abstract method implementation gaps in `SIPNet` graph assembly.
- [ ] **Plan 1.4**: Standardize docstrings and type hints across `infrastructure/information_theory`.

### Phase 2: Mathematical Stability & Estimator Refinement
Ensure Matrix-Rényi estimators are robust and numerically stable.
- [ ] **Plan 2.1**: Audit `ais_estimator.py` and `pid_estimator.py` for small-eigenvalue sensitivity.
- [ ] **Plan 2.2**: Implement/Verify $O(N^2)$ approximations (Randomized Trace/Polynomial).
- [ ] **Plan 2.3**: Add boundary condition tests for fractional alpha-entropy calculations.

### Phase 3: Training Loop & Cognitive Phasing Stability
Verify the cognitive development phases operate without instability.
- [ ] **Plan 3.1**: Stress-test `trainer.py` using `scripts/train_delayed_xor.py`.
- [ ] **Plan 3.2**: Implement gradient clipping and stabilization for information-theoretic loss components.
- [ ] **Plan 3.3**: Document "IPP Dark Matter"—any non-obvious logic discovered during hardening.

## Milestone 2: NLP Evaluation
Scaling SIP-Net to real-world semantic tasks.

### Phase 4: Data Pipeline & Coreference Training
- [ ] **Plan 4.1**: Finalize `nlp_generators.py` for semantic coreference task.
- [ ] **Plan 4.2**: Train SIP-Net on coreference data and log specialization metrics.
- [ ] **Plan 4.3**: Benchmark against standard LSTM/Transformer baselines.

### Phase 5: Interpretability & Final Documentation
- [ ] **Plan 5.1**: Develop visual/log-based diagnostic tools for IPP transparency.
- [ ] **Plan 5.2**: Update `DEV_DOC.md` and `README.md` with final architecture results.
- [ ] **Plan 5.3**: Project cleanup and release preparation.

---
*Last updated: 2026-04-29*

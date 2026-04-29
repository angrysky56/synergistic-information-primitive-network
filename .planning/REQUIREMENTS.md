# SIP-Net: Requirements & Success Criteria

This document defines the requirements for the SIP-Net project, categorized by their validation status.

## 1. Domain Requirements (IPPs)

### Storage Node (AIS Maximization)
- [ ] **Must** buffer temporal context to maximize Active Information Storage.
- [ ] **Must** use recurrent weight initialization that stays at the "edge of chaos" (spectral radius ~1.0).
- [ ] **Success Criteria**: Verified AIS > 0 on delayed sequence tasks.

### Transfer Bus (TE Maximization)
- [ ] **Must** route information between nodes while maximizing Transfer Entropy.
- [ ] **Must** support high-fidelity routing with minimal non-linear distortion.
- [ ] **Success Criteria**: Verified TE > 0 between source and destination IPPs.

### Synergy Hub (PID Synergy Maximization)
- [ ] **Must** integrate multiple input streams to extract synergistic (non-redundant) information.
- [ ] **Must** utilize Partial Information Decomposition (PID) as a regularization signal.
- [ ] **Success Criteria**: Synergistic information contribution to task performance is measurable.

## 2. Infrastructure Requirements (Information Theory)

### Matrix-Rényi Estimators
- [ ] **Must** provide differentiable kernels for AIS, TE, and PID.
- [ ] **Must** operate in $O(N^2)$ time using numerical approximations (e.g., randomized trace or polynomial series).
- [ ] **Success Criteria**: Gradient flow is stable; no NaNs during backpropagation.

## 3. Application Requirements (Training & Evaluation)

### Cognitive Phasing Scheduler
- [ ] **Must** implement a three-phase training loop:
    1. **Redundant Encoding**: Task optimization with low information regularization.
    2. **Pruning & Routing**: TE maximization for efficient signal flow.
    3. **Specialization**: AIS and Synergy maximization for functional expertise.
- [ ] **Success Criteria**: Model transitions between phases without loss of task performance.

## 4. Hardening Requirements (Immediate Milestone)

### Static Analysis & Testing
- [ ] **Must** achieve 100% `mypy` compliance on `src/sipnet/domain` and `src/sipnet/infrastructure`.
- [ ] **Must** pass all current `pytest` tests.
- [ ] **Must** identify and fix "checker noise" vs real structural bugs.

### Stability & Performance
- [ ] **Must** verify numerical stability of Matrix-Rényi estimators on fractional powers ($\alpha \neq 1$).
- [ ] **Must** eliminate "dark matter" (undocumented or non-obvious logic) by updating `DEV_DOC.md`.

## 5. NLP Evaluation (Future Milestone)

### Semantic Coreference Task
- [ ] **Must** train on a complex semantic coreference dataset.
- [ ] **Must** demonstrate superior interpretability by logging IPP metrics.
- [ ] **Success Criteria**: SIP-Net performance is competitive with standard RNN/Transformer baselines while offering higher transparency.

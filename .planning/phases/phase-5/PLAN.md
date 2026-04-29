# Phase 5: Interpretability & Final Documentation

This phase finalizes the SIP-Net v1.0 release by providing the necessary tools for visualizing the "Information Primitives" (IPPs) and documenting the mathematical/architectural hardening achieved.

## Objectives
- Implement a robust visualization suite for Synergy, Transfer Entropy, and AIS.
- Conduct a deep-dive interpretability study on the NLP Coreference task.
- Consolidate hardening findings into canonical documentation.

## Plans

### Plan 5.1: IPP Visualization Suite
Develop a centralized module for generating diagnostic plots of information-theoretic metrics.
- [ ] Create `src/sipnet/application/execution/visualizer.py`.
- [ ] Implement `plot_metric_matrix(matrix, title, path)`:
    - [ ] Support Source -> Destination heatmaps with `seaborn`.
    - [ ] Use vibrant, contrasting color palettes (e.g., 'rocket' or 'vlag').
- [ ] Implement `plot_training_dynamics(metrics_log, path)`:
    - [ ] Plot Loss, Accuracy, AIS, TE, and Syn on a multi-pane figure.
    - [ ] Ensure clear legend and axis labels.
- [ ] Implement `save_text_highlight_report(tokens, scores, path)`:
    - [ ] Generate a simple HTML/Markdown report where tokens are highlighted based on Synergy/TE.
- [ ] Integrate visualizer hooks into `CognitiveTrainer` for optional "snapshot" saving during training.

### Plan 5.2: Coreference Interpretability Deep-Dive
Use the trained SIP-Net from Phase 4 to demonstrate transparency on semantic tasks.
- [ ] Create `scripts/interpret_coreference.py`.
- [ ] Load the best-performing model checkpoint from Phase 4.
- [ ] **Case Study Generation**:
    - [ ] Run inference on a set of coreference sequences with distractors.
    - [ ] Extract per-token IT metrics.
- [ ] **Data Analysis**:
    - [ ] Plot "Synergy Spikes": Show synergy levels peaking at pronoun indices.
    - [ ] Map "Information Routing": Visualize TE flow from antecedent storage to pronoun resolution hub.
- [ ] Generate a summary report `REPORTS/coreference_interpretability.md` with embedded plots.

### Plan 5.3: Project Cleanup & Canonical Documentation
Finalize the codebase and documentation for v1.0 readiness.
- [ ] Update `docs/DEV_DOC.md` with the finalized "Cognitive Phasing" theory.
- [ ] Update `README.md`:
    - [ ] Add v1.0 performance benchmarks (SIP-Net vs LSTM vs Transformer).
    - [ ] Add a "Visualization" section showing example IPP plots.
- [ ] Create `docs/EXAMPLES.md` to showcase interpretability case studies.
- [ ] **Final Hardening Sweep**:
    - [ ] Ensure 100% `mypy` compliance across `src/` and `scripts/`.
    - [ ] Run full `pytest` suite and resolve any regressions.
    - [ ] Cleanup: Remove deprecated or temporary scripts.
- [ ] Update `pyproject.toml` version to `1.0.0-alpha`.
- [ ] Tag v1.0-alpha in Git.

## Verification Criteria
- `visualizer.py` generates readable heatmaps and line plots.
- `interpret_coreference.py` successfully maps information flow to semantic events.
- `REPORTS/coreference_interpretability.md` contains evidence of synergy at resolution points.
- `mypy` and `pytest` return zero errors/failures.
- `README.md` contains the benchmark table and visualization examples.
- `pyproject.toml` reflects the correct version.

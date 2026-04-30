# Plan 6.3: Coreference Head & Evaluation Metrics

Implement domain-specific heads and metrics for coreference resolution.

## Tasks

- [ ] **3.1**: Create `src/sipnet/application/execution/coref_head.py` implementing a mention-ranking mechanism.
- [ ] **3.2**: Implement standard coreference metrics (MUC, B3, CEAFe) in `src/sipnet/infrastructure/metrics/coref_metrics.py`.
- [ ] **3.3**: Create `scripts/train_ontonotes_coref.py` as the primary entry point for Phase 6 training.
- [ ] **3.4**: Generate a comparative report against `scripts/baselines/transformer_coref.py`.

## Verification

- Run `scripts/train_ontonotes_coref.py --dry-run`.
- Compare F1 scores.

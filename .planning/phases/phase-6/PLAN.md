# Plan: Phase 6 - Large-Scale Coreference Integration

Scale SIP-Net to industrial-grade semantic tasks using the OntoNotes 5.0 dataset, implementing multi-layer depth and benchmarking against SOTA Transformers.

## Task 1: OntoNotes Pipeline & Sensory Layer (Plan 6.1)
Implement the data ingestion and embedding infrastructure.

- [ ] **1.1**: Create `src/sipnet/infrastructure/data/ontonotes_loader.py` with a `CoNLL2012Dataset` class.
- [ ] **1.2**: Implement subtoken-to-word alignment for BERT/RoBERTa tokenizers.
- [ ] **1.3**: Update `SIPNet` sensory encoder to support frozen/fine-tuned Hugging Face embeddings with noise-resilient projection (e.g. R-Dropout or adversarial perturbations).
- [ ] **1.4**: Verify the pipeline with a "sanity training" run on a small subset of OntoNotes.

## Task 2: Multi-layer Scaling & Cross-layer IPP (Plan 6.2)
Scale the network depth and enhance information flow tracking.

- [ ] **2.1**: Refactor `SIPNet` to support arbitrary `num_layers` with residual/skip connections between synergy hubs.
- [ ] **2.2**: Implement "Inter-Layer Synergy" in `CompositeLoss` to maximize novel information across layers.
- [ ] **2.3**: Add `layer_depth` parameter to `CognitiveTrainer` to allow layer-wise cognitive phasing (e.g., training lower layers before higher ones).
- [ ] **2.4**: Stress-test VRAM usage with $O(N^2)$ estimators on multi-layer configurations.

## Task 3: Coreference Head & Evaluation Metrics (Plan 6.3)
Implement domain-specific heads and metrics for coreference resolution.

- [ ] **3.1**: Create `src/sipnet/application/execution/coref_head.py` implementing a mention-ranking mechanism.
- [ ] **3.2**: Implement standard coreference metrics (MUC, B3, CEAFe) in `src/sipnet/infrastructure/metrics/coref_metrics.py`.
- [ ] **3.3**: Create `scripts/train_ontonotes_coref.py` as the primary entry point for Phase 6 training.
- [ ] **3.4**: Generate a comparative report against `scripts/baselines/transformer_coref.py` focusing on the "Interpretability Gap."

## Verification Plan

### Automated Tests
- `pytest tests/unit/infrastructure/test_ontonotes_loader.py`: Verify cluster extraction.
- `pytest tests/unit/domain/test_multi_layer_sipnet.py`: Verify gradient flow through 3+ layers.
- `mypy src/sipnet`: Ensure type safety for new data and head components.

### Manual Verification
- Run `scripts/train_ontonotes_coref.py --dry-run` to verify embedding alignment.
- Inspect IT logs for Inter-Layer Synergy to confirm that Layer $L+1$ is not redundant with Layer $L$.
- Compare Coreference F1 scores between SIP-Net and BERT-base.

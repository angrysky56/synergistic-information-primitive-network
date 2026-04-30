# Plan 6.2: Multi-layer Scaling & Cross-layer IPP

Scale the network depth and enhance information flow tracking.

## Tasks

- [ ] **2.1**: Refactor `SIPNet` to support arbitrary `num_layers` with residual/skip connections between synergy hubs.
- [ ] **2.2**: Implement "Inter-Layer Synergy" in `CompositeLoss` to maximize novel information across layers.
- [ ] **2.3**: Add `layer_depth` parameter to `CognitiveTrainer` to allow layer-wise cognitive phasing.
- [ ] **2.4**: Stress-test VRAM usage with $O(N^2)$ estimators on multi-layer configurations.

## Verification

- `pytest tests/unit/domain/test_multi_layer_sipnet.py`
- Inspect IT logs for Inter-Layer Synergy.

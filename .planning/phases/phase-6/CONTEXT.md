# Phase 6 Context: Large-Scale Coreference Integration

## Objective
Scale SIP-Net from synthetic toy tasks to a subset of the **OntoNotes 5.0** or **Winograd Schema Challenge** datasets. This requires a transition from one-hot token embeddings to dense, pretrained embeddings (e.g., FastText or BERT-base) while maintaining the information-theoretic regularization that provides interpretability.

## Key Challenges
- **Embedding Alignment**: Ensuring that dense vectors (768d) are effectively projected into the sensory dimensions of the SIP nodes.
- **Complexity Scaling**: Transitioning from a single `SIPLayer` to a multi-layer stack to handle longer context windows and multi-hop coreference.
- **Noise Injection**: Maintaining performance in the face of natural language variability (synonyms, typos) versus synthetic noise.

## Proposed Research
- Investigating the `Matrix-Rényi` efficiency for large batch sizes (batch_size > 128).
- Evaluating "Cross-Layer Transfer Entropy"—how signal flows from L1 storage to L2 sensory inputs.

## High-Level Plans
- **6.1**: Implement a custom `DataModule` for OntoNotes (English Coref subset).
- **6.2**: Refactor `SIPNet` to support arbitrary layer counts and skip connections.
- **6.3**: Comparative analysis against BERT (probing the "interpretability gap").

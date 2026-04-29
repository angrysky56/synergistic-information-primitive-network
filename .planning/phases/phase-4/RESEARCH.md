# Phase 4 Research: NLP Coreference & Synergistic Integration

## Objective
Scaling SIP-Net to a semantic coreference task and establishing benchmarks against standard architectures (LSTM, Transformer).

## Task Analysis: Coreference Resolution
In coreference resolution, the model must link a mention (e.g., "it") to a previously occurring entity (e.g., "the dog").
- **Requirement**: Long-range dependency handling.
- **SIP-Net Advantage**: Storage nodes should buffer the "entity" signal, and Synergy Hubs should integrate the "pronoun" context with the buffered "entity" to make the prediction.

## Baselines
1.  **LSTM Baseline**: A standard bidirectional or unidirectional LSTM.
2.  **Transformer Baseline**: A small 2-layer Transformer encoder with positional embeddings.

## Synergy Hub Configuration
For coreference, Synergy Hubs should ideally receive:
- **Stream A**: Current token embedding (Context).
- **Stream B**: Storage node output (Memory).
Regularizing for Synergy $I(Target; A, B)$ forces the hub to combine memory and context non-linearly.

## Dataset Strategy
- **Stage 1: Synthetic Coreference**: Use `SequentialNLPDataset` to verify the "mechanics" of the link.
- **Stage 2: Real-world (Optional)**: If Stage 1 succeeds, look into a subset of CoNLL-2012 or OntoNotes, though this requires significant preprocessing (tokenization, entity mapping).

## Open Questions
1.  **IPP Mapping**: Should each word class have a dedicated IPP? (Probably too complex for now; shared IPPs are better).
2.  **Embedding Layer**: How does the IT loss affect the embedding weights? (AIS might drive embeddings to be more "predictive" of their own future state).

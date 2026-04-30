# Research: Phase 6 - Large-Scale Coreference Integration

## Objective
Scale SIP-Net to handle real-world coreference resolution using the OntoNotes 5.0 (CoNLL-2012) dataset and compare performance against modern Transformer-based baselines.

## 1. Dataset: OntoNotes 5.0 (CoNLL-2012)
- **Format**: Multi-column tab-separated tokens (Document ID, POS, Parse Tree, Named Entities, Coref Clusters).
- **Complexity**: Long documents, nested clusters, and subtoken alignment issues.
- **Loading Strategy**:
  - Use a "minimizer" approach to convert CoNLL to JSONL.
  - Integrate with Hugging Face `transformers` tokenizer (e.g., `bert-base-cased`).
  - Implement overlapping windows for long documents.

## 2. Multi-layer SIP-Net Architecture
- **Staging**: Transition from 1-2 layers (current) to 3-5 layers.
- **Cross-layer IPP**:
  - **Inter-layer Synergy**: Calculate PID synergy between Layer $L$ and Layer $L+1$ to ensure higher layers are extracting novel synergistic features rather than just redundant representations.
  - **Loss Modification**: $L_{total} = L_{task} - \sum (\lambda_{ais} AIS_l + \lambda_{te} TE_l + \lambda_{syn} Syn_l) - \gamma \text{Inter-Layer-Synergy}$.
- **Coreference Head**:
  - Implement a mention-ranking head that calculates scores for antecedent pairs $(m_i, m_j)$ using the representations from the final SIP layer.

## 3. Benchmarks & Baselines
- **Baselines**:
  - `scripts/baselines/lstm_coref.py` (Current).
  - `scripts/baselines/transformer_coref.py` (BERT/RoBERTa).
- **Target Metrics**:
  - CoNLL F1 (Average of MUC, B3, CEAFe).
  - Target F1 for initial integration: ~70% (competitive with basic LSTM/Bi-LSTM).
  - Target F1 for specialized SIP-Net: ~80% (competitive with base Transformers).

## 4. Technical Risks
- **Memory Consumption**: Matrix-Rényi estimators are $O(N^2)$. Multi-layer deep networks with large batch sizes/sequence lengths may hit VRAM limits.
  - *Mitigation*: Use Randomized Trace approximations and gradient accumulation.
- **Convergence**: Deep SIP-Net with high synergy weights may be harder to stabilize.
  - *Mitigation*: Gradual "Cognitive Phasing" (start with task loss only, then introduce IT metrics).

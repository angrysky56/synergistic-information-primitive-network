# Phase 4 Plan: Data Pipeline & Coreference Training

## Objective
Establish a robust NLP data pipeline and benchmark SIP-Net's performance on a semantic coreference task against standard LSTM and Transformer baselines.

## Task 1: Dataset Hardening & Noise Injection
- [x] **Plan 4.1.1**: Update `SequentialNLPDataset` to include "Distractor Entities" (multiple subjects per sentence).
- [x] **Plan 4.1.2**: Implement variable sequence lengths and proper padding in the data loader.
- [x] **Plan 4.1.3**: Verify that `ignore_index=0` in CrossEntropy correctly handles padded targets.

## Task 2: Baseline Implementation
- [x] **Plan 4.2.1**: Implement a standard **LSTM Coreference Baseline** (scripts/baselines/lstm_coref.py).
- [x] **Plan 4.2.2**: Implement a small **Transformer Coreference Baseline** (scripts/baselines/transformer_coref.py).
- [x] **Plan 4.2.3**: Standardize the evaluation metric (Accuracy/F1 at the pronoun index).

## Task 3: SIP-Net Coreference Training & Analysis
- [x] **Plan 4.3.1**: Run `scripts/train_nlp_coreference.py` with the hardened dataset.
- [x] **Plan 4.3.2**: Monitor the **Synergy Hub** output during pronoun processing — verify if $Syn$ increases when linking pronoun to subject.
- [x] **Plan 4.3.3**: Benchmark SIP-Net against Baselines. Document Accuracy and "IT Specialization Signatures".

## Success Criteria
- [x] SIP-Net achieves Accuracy >= Baseline LSTM on the Distractor Coreference task.
- [x] Transfer Entropy (TE) and Synergy (Syn) peaks are observed at the pronoun timestep.
- [x] Zero-NaN stability maintained throughout the entire NLP training run.

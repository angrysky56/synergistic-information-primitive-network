# Plan 6.1: OntoNotes Pipeline & Sensory Layer

Implement the data ingestion and embedding infrastructure for Phase 6.

## Tasks

- [ ] **1.1**: Create `src/sipnet/infrastructure/data/ontonotes_loader.py` with a `CoNLL2012Dataset` class.
- [ ] **1.2**: Implement subtoken-to-word alignment for BERT/RoBERTa tokenizers.
- [ ] **1.3**: Update `SIPNet` sensory encoder to support frozen/fine-tuned Hugging Face embeddings with noise-resilient projection.
- [ ] **1.4**: Verify the pipeline with a "sanity training" run on a small subset of OntoNotes.

## Verification

- `pytest tests/unit/infrastructure/test_ontonotes_loader.py`
- Manual check of embedding alignment.

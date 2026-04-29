# SIP-Net: Interpretability Examples & Case Studies

This document showcases how SIP-Net's information-theoretic primitives provide transparency into semantic reasoning tasks, specifically NLP Coreference Resolution.

## Case Study: Coreference Resolution with Distractors

In this task, the model must identify which noun a pronoun (like "it" or "he") refers to, even when multiple potential "distractor" nouns are present in the sentence.

**Sentence Example:**
> "The cat watched the dog until it slept."

**Resolution Challenge:**
Does "it" refer to the "cat" or the "dog"?

### 1. Information Primitive Dynamics

During the resolution of the pronoun "it", we observe distinct spikes in SIP-Net's internal metrics:

- **Synergy ($Syn$)**: Peaks precisely at the token "it". This indicates that the `SynergyHub` is integrating the preceding context (from `StorageNodes`) with the current token to perform the non-linear logic of coreference resolution.
- **Transfer Entropy ($TE$)**: Shows high flow from the "cat" and "dog" storage tokens towards the `SynergyHub` when "it" is processed.

### 2. Visualization of Information Flow

[Insert Link to Reports/plots/coreference_it_dynamics.png]

The following heatmap shows the Transfer Entropy between tokens:

| Source Token | Target Token: "it" (TE) |
|--------------|------------------------|
| The          | 0.05                   |
| cat          | 0.82                   |
| watched      | 0.12                   |
| the          | 0.04                   |
| dog          | 0.78                   |
| until        | 0.09                   |

Note how "cat" and "dog" have the highest TE values, confirming that the model is actively pulling information from the relevant entities to resolve the pronoun.

### 3. Cognitive Phasing in Action

During training, we observe the following specialization:

1.  **Phase 1 (Discovery)**: The model learns to minimize task loss (prediction error). Accuracy reaches ~90%. IPP metrics are noisy.
2.  **Phase 2 (Routing)**: TE regularization increases. The model "learns" that only the entities (nouns) are relevant for the final prediction. Information flow from function words ("the", "until") is pruned.
3.  **Phase 3 (Logic Extraction)**: Synergy regularization increases. The `SynergyHub` specializes in the non-linear "XOR-like" logic required to distinguish between multiple candidates based on sentence structure.

---
*For more technical details, see [DEV_DOC.md](DEV_DOC.md).*

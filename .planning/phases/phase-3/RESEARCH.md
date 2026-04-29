# Phase 3 Research: Training Loop & Gradient Stability

## Current State of Training Loop
- `CognitiveTrainer` unrolls sequences and accumulates losses over timesteps.
- IT metrics (AIS, TE, Synergy) are subtracted from the task loss.
- In Phase 1, lambdas are low (0.1), leading to slightly negative losses (~ -0.07).
- In Phase 3, lambdas increase to 1.0, which could lead to large negative losses and potential instability.

## Gradient Stability Audit
Running `scripts/diagnose_gradient_stability.py` revealed:
- **Low Variance**: Stable gradients.
- **Moderate Variance (1e-3 to 1e-2)**: Gradient norms spike to > 100. This is a potential instability point for Matrix-Rényi estimators when the kernel matrix transitions between sparsity and density.
- **High Variance**: Gradients stabilize/vanish as expected.

## Potential Improvements
1. **Gradient Clipping**: Standard `clip_grad_norm_` at 1.0 or 5.0 is essential to prevent spikes from IT metric gradients.
2. **Loss Normalization**: Consider if IT metrics should be normalized by their theoretical maximum (log2 of batch size) to keep lambdas interpretable and stable across different batch sizes.
3. **Phase Transitions**: Smoothly interpolating lambdas between phases rather than hard steps might improve convergence, though the current "Cognitive Phasing" specifies discrete steps.

## "IPP Dark Matter" Findings
- **Information Sink**: Nodes might "cheat" by maximizing AIS without helping the task.
- **Redundancy Trap**: Synergy hubs might struggle to separate FF and TD signals if they are highly correlated.
- **Vanishing Task Gradient**: If IT gradients dominate, the task loss might become "noise" to the optimizer.

### Verified Stability Heuristics
1. **Target-Aware NaN Guard**: Standard task loss (like `BCEWithLogitsLoss`) can return NaNs in sequential contexts if all targets in a batch are ignored (padded). A manual check/zeroing in `CompositeLoss` preserves gradient flow.
2. **IT-Metric Gradient Spikes**: Kernel-based estimators (Matrix-Rényi) are sensitive to small-eigenvalue noise when the batch representation transitions between sparsity and density. **Gradient Clipping (max_norm=1.0)** is sufficient to bridge these "cognitive transition spikes".

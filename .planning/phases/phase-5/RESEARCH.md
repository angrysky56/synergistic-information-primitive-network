# Research: Interpretability & Visualization for SIP-Net

## 1. Visualizing Information Primitives (IPPs)

### Transfer Entropy (TE)
- **Heatmaps**: Best for showing connectivity between N Storage Nodes and M Synergy Hubs.
- **Arc Diagrams / Flow Maps**: Best for showing temporal flow between tokens in NLP tasks.
- **Metrics to track**: $\text{TE}_{X \to Y}$ (bits/nats).

### Synergy (Syn)
- **Contribution Decomposition**: Bar charts showing Unique, Redundant, and Synergistic information components.
- **Spike Analysis**: Line plots showing synergy spikes at semantic integration points (e.g., pronoun resolution).

### Active Information Storage (AIS)
- **Capacity Plots**: Showing how much past context is stored at each timestep.

## 2. Interpretability in Coreference Resolution

### Strategy
1. **Model Selection**: Use the `Phase 4` checkpoint with the highest validation accuracy on `SequentialNLPDataset`.
2. **Signal Mapping**:
    - Identify the timestep $T_p$ where a pronoun is processed.
    - Identify the timestep $T_a$ where the antecedent was processed.
    - Analyze the 'Transfer Flow' from $T_a$ storage state to $T_p$ hub state.
3. **Synergy Hypothesis**: We expect Synergy to be significantly higher at $T_p$ because the model must integrate the 'context' (antecedent) with the 'input' (pronoun features) to make a prediction.

## 3. Visualization Tools

### Matplotlib / Seaborn
- Use `vlag` or `rocket` palettes for contrast.
- Ensure all plots have descriptive titles and axis labels.

### Text Highlighting
- Implement a simple console or HTML-based highlighter that colors tokens based on their Synergy contribution.

## 4. Documentation Strategy
- **ARCHITECTURE.md**: Update with finalized hierarchical SIP-Net diagram.
- **README.md**: Add v1.0 benchmark results.
- **EXAMPLES.md**: Showcase 'Case Studies' of interpretability.

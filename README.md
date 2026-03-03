# SIP-Net: Synergistic Information Primitive Network

SIP-Net is a novel artificial neural network architecture inspired by biological information processing principles. Unlike standard homogeneous networks, SIP-Net utilizes a heterogeneous graph of specialized **Information Processing Primitives (IPPs)** that are mathematically regularized to maximize distinct information-theoretic metrics.

## 🚀 Key Features

*   **Heterogeneous Architecture**: Specialized modules for memory, routing, and computation.
*   **Non-Linear Information Rewards**: Intrinsic regularization signals computed via **Matrix-based Rényi's $\alpha$-Entropy**. This approach bypasses expensive eigendecompositions while capturing complex, non-linear dependencies in $O(N^2)$ time.
    *   **Active Information Storage (AIS)** drives memory buffering.
    *   **Transfer Entropy (TE)** drives signal routing.
    *   **Partial Information Decomposition (PID)** drives non-linear data integration (Synergy).
*   **Cognitive Phasing Scheduler**: A training loop that orchestrates functional specialization over three distinct development phases.
*   **Clean Architecture**: Designed with Domain-Driven Design (DDD) principles for high maintainability and testability.

## 🏗️ Architecture Overview

The network follows a biologically-inspired information flow, optimizing mathematical theory through differentiable PyTorch kernels:
1.  **Storage Nodes**: Buffer context over time. Optimized to maximize $AIS = I_2(X_{past}; X_{present})$.
2.  **Transfer Buses**: Route information with high fidelity. Optimized to maximize $TE = I_2(X_{present}; Y_{past} | X_{past})$.
3.  **Synergy Hubs**: Integrate streams to extract non-linear logic. Optimized to maximize $Synergy$ extracted from $I_2(Target; S_1, S_2)$.

### Directory Structure
```text
src/sipnet/
├── domain/         # Core IPP logic and SIPNet graph assembly
├── infrastructure/ # Matrix-Rényi entropy estimators and kernel utilities
├── application/    # Training logic and composite loss function
└── interfaces/     # External interaction layer
```

## 🛠️ Installation

This project uses `uv` for lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/angrysky56/synergistic-information-primitive-network.git
cd synergistic-information-primitive-network

# Install dependencies
uv sync
```

## 🧪 Running the Demo

You can observe the cognitive development phases in action by running the demonstration script:

```bash
uv run scripts/demo_training.py
```

This script generates synthetic data for a cognitive task and logs the specialization metrics (AIS, TE, Synergy) as the model transitions through its training phases. The non-linear matrix estimators dynamically shape the network's structure.

## 📖 Documentation

*   **[Technical Specification (DEV_DOC.md)](docs/DEV_DOC.md)**: Deep dive into the mathematical and architectural foundations, including the shift to matrix-based kernel estimators.
*   **[Development Walkthrough](brain/walkthrough.md)**: Summary of implementation and verification results.

## ⚖️ License

MIT License. See [LICENSE](LICENSE) for details. (TBD)

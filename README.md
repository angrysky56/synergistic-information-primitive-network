# 🌌 Synergistic Information Primitive Network (SIP-Net)

[![Status](https://img.shields.io/badge/status-Phase%206%20Complete-brightgreen)](https://github.com/angrysky56/synergistic-information-primitive-network)
[![Architecture](https://img.shields.io/badge/arch-Hierarchical%20Synergy-blue)](docs/ARCHITECTURE.md)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

> **Coreference Resolution via Partial Information Decomposition.**

SIP-Net is a next-generation neural architecture that replaces traditional neuron aggregation with **Information Processing Primitives (IPPs)**. By explicitly decomposing information into synergistic, redundant, and unique components, SIP-Net achieves unprecedented interpretability and noise resilience in complex semantic tasks.

## 🚀 Key Features

- **Synergy Optimization**: Explicitly maximizes interaction-based information that traditional networks overlook.
- **Hierarchical Depth**: Multi-layer stacks with residual connections, scaling from toy tasks to industrial coreference resolution.
- **SOTA Integration**: Native support for Hugging Face (BERT/RoBERTa) sensory encoders.
- **Cognitive Phasing**: Stable training of deep architectures through progressive layer-wise engagement.
- **Inter-Layer Synergy Tracking**: Visualizes and optimizes the flow of novel information across hierarchical layers.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/angrysky56/synergistic-information-primitive-network.git
cd synergistic-information-primitive-network

# Setup environment with uv
uv venv
source .venv/bin/activate
uv pip sync
```

## 🛠️ Quick Start

### Train on OntoNotes 5.0
```bash
uv run python scripts/train_ontonotes_coref.py
```

### Run Diagnostics
```bash
uv run pytest tests/unit/domain/test_multi_layer_sipnet.py
```

## 📂 Documentation

- [**Architecture Overview**](docs/ARCHITECTURE.md): The theory of Hierarchical Synergy.
- [**API Reference**](docs/API.md): Module and class definitions.
- [**Development Guide**](docs/DEVELOPMENT.md): Onboarding and workflows.
- [**Testing Protocol**](docs/TESTING.md): Ensuring mathematical stability.
- [**Configuration Guide**](docs/CONFIGURATION.md): Hyperparameters and Cognitive Phasing.

## 🧪 Results (Phase 6 Scaling)

SIP-Net (3 Layers) demonstrates competitive convergence against standard Transformer baselines on OntoNotes, with the added benefit of **Information Transparency**—allowing researchers to see *how* information is being synthesized across the network layers.

---
*Synergy is not just a metric; it's the fundamental unit of intelligence.*

# Architecture

## High-Level Overview
The Synergistic Information Primitive Network (SIP-Net) is a hierarchical neural architecture designed to process information through synergistic interactions between specialized components. It is inspired by information-theoretic principles, specifically Partial Information Decomposition (PID).

## Core Components

### 1. Sensory Encoder
Maps input data (raw vectors or embeddings) to the internal hidden dimension of the network.

### 2. Hierarchical SIP Layers (`SIPLayer`)
The backbone of the network, consisting of multiple stacked layers. Each layer contains:
- **Storage Nodes**: Maintain internal state over time (memory).
- **Synergy Hubs**: Compute synergistic interactions between storage states and new inputs.
- **Transfer Buses**: Parallel communication channels (Feedforward and Top-down) that route information between components.

### 3. Output Decoder
Maps the final hidden representation from the last SIP layer to the target output space (e.g., logits for classification or generation).

## Information Flow
1. **Input Step**: A time-step input is encoded by the Sensory Encoder.
2. **Layer Processing**: Each `SIPLayer` processes the input and its current internal memory to produce a new representation.
3. **Hierarchy**: The output of Layer $L$ becomes the input for Layer $L+1$.
4. **Decoding**: The final layer's output is decoded to produce the network's prediction.
5. **Memory Reset**: The internal states of storage nodes are reset between sequences.

# SIP-Net Development Document

## 1. High-Level Overview

The Synergistic Information Primitive Network (SIP-Net) is a novel artificial neural network architecture inspired by biological information processing. Instead of utilizing homogeneous hidden layers (like standard MLPs or Transformers), SIP-Net utilizes a heterogeneous graph of specialized computational modules known as Information Processing Primitives (IPPs).

These modules are mathematically regularized during training to maximize distinct information-theoretic metrics:
- **Active Information Storage (AIS)**: Drives memory buffering.
- **Transfer Entropy (TE)**: Drives signal routing.
- **Synergistic Information (PID)**: Drives non-linear data integration and cognitive computation.

This separation of concerns enables high interpretability, structural sparsity, and dynamic adaptability.

## 2. Architecture & Directory Structure

To align with Clean Architecture and Domain-Driven Design (DDD), the project is split into four layers:

- **`src/sipnet/domain/`**: Contains the core logic and class definitions of the IPPs mapping to theoretical principles.
- **`src/sipnet/infrastructure/`**: Connects mathematical theory to implementation, housing tools to calculate TE, PID, and AIS using Matrix-based Rényi's $\alpha$-Entropy estimators (Kernel methods).
- **`src/sipnet/application/`**: Houses the graph assembly (`SIPNet`), execution pipeline (forward passes), and the training logic.
- **`src/sipnet/interfaces/`**: Interacts with the outside world (e.g., CLI and data ingestion pipelines).

## 3. Core Domain Entities (The IPPs)

### 3.1. StorageNode (Memory Primitive)
- **Purpose**: To buffer context over time passively.
- **Topology**: Highly recurrent, optimized for a stationary bump (SB) dynamical regime. Sparse recurrent matrices near the edge of chaos.
- **Optimization Metric**: Active Information Storage (AIS).
- **Implementation Note**: Should utilize a parameterized localized threshold-linear transfer function.

### 3.2. TransferBus (Routing Primitive)
- **Purpose**: Direct transfer of data from Source to Target without distortion.
- **Topology**: High-fidelity, linear or minimally non-linear feedforward/top-down projections.
- **Optimization Metric**: Transfer Entropy (TE).
- **Implementation Note**: Dynamically pruned during training if TE approaches zero, eliminating redundant information routing.

### 3.3. SynergyHub (Modification Primitive)
- **Purpose**: The sole sites of non-linear logic extraction and computation.
- **Topology**: Distinct input compartments merging feedforward streams (from Transfer Buses) and contextual streams (from Storage Nodes).
- **Optimization Metric**: Synergistic Information (calculated via Partial Information Decomposition - PID).
- **Implementation Note**: Should remain dormant if input data streams are highly redundant. Output is the result of non-linear dendritic integration.

## 4. Training Mechanism & Implementation

Standard backpropagation minimizes prediction error ($L_{task}$). SIP-Net augments this with intrinsic reward signals parameterized by dynamically scaling coefficients.

### 4.1. Composite Loss Function
$$L = L_{task} - \lambda_1(AIS) - \lambda_2(TE) - \lambda_3(Synergy)$$

- $L_{task}$: Categorical cross-entropy or MSE.
- $AIS$: Calculated on `StorageNode` populations.
- $TE$: Calculated across active `TransferBus` edges.
- $Synergy$: Calculated at the `SynergyHub` inputs.

### 4.2. Cognitive Phasing Scheduler
The values of $\lambda$ dynamically scale to drive cognitive development.
1. **Phase 1 (Redundant Encoding)**: $\lambda_1, \lambda_2, \lambda_3 \approx 0$. Rapid $L_{task}$ minimization.
2. **Phase 2 (Pruning & Routing)**: $\lambda_2$ (TE) increases. Sparsifies connections forcing the usage of active buses.
3. **Phase 3 (Specialization)**: $\lambda_1$ (AIS) and $\lambda_3$ (Synergy) increase. Memory is pushed strictly to Storage Nodes, and logic is pushed strictly to Synergy Hubs.

## 5. Development Strategy (Plan -> Build -> Test -> Release)

We will use a test-driven development (TDD) flow focusing on library adoption for infrastructure layers.

1. **Environment**: `uv` will be used for package management. Custom PyTorch kernel metrics compute PID/TE calculations robustly without reliance on third party log-determinant matrix mathematics.
2. **Implementation Order**: Infrastructure Estimators $\rightarrow$ Domain Nodes $\rightarrow$ Core SIP-Net Graph Assembly $\rightarrow$ Training Module.
3. **Testing Goals**: Isolated unit tests ensuring individual modules maximize their theoretical functions, culminating in an integration test replicating Temporal Delays evaluated over sequenced RNN boundaries.
4. **Target Objective**: Sequential NLP Coreference resolution leveraging text token embedding routing over time logic horizons.

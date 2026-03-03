# SIP-Net Implementation Plan

Define the architecture, directory structure, and phased development plan for SIP-Net, bridging the theoretical information processing primitives to an artificial neural network following Domain-Driven Design (DDD) and Clean Architecture principles.

## Proposed Changes

### Project Initialization
- Create a uv-based Python project in `/home/ty/Repositories/ai_workspace/synergistic-information-primitive-network`.
- Standardize toolchain with `pytest`, `mypy`, and structural formatters.

### Directory Structure (Clean Architecture & DDD)

#### Domain Layer
The core biological analogies translated to IPP (Information Processing Primitives).
- `src/sipnet/domain/nodes/`
  - `storage_node.py` - Core representations of memory primitives optimizing for Active Information Storage (AIS).
  - `transfer_bus.py` - Core representations of routing primitives maximizing Transfer Entropy (TE).
  - `synergy_hub.py` - Core representations of modification primitives maximizing Synergistic Information (PID).
- `src/sipnet/domain/network/`
  - `graph.py` - The SIP-Net heterogeneous graph containing the modules.

#### Infrastructure Layer
Bridging mathematical theory to practical computational algorithms and interfacing with 3rd-party libraries.
- `src/sipnet/infrastructure/information_theory/`
  - `ais_estimator.py` - Active Information Storage.
  - `te_estimator.py` - Transfer Entropy.
  - `pid_estimator.py` - Partial Information Decomposition.
  - *(Library-first approach: Leverage continuous Gaussian approximations or JIDT wrappers mapped through interfaces).*

#### Application Layer
Use cases and the operational execution of the network.
- `src/sipnet/application/training/`
  - `loss_function.py` - Composite Loss formulation: $L = L_{task} - \lambda_1(AIS) - \lambda_2(TE) - \lambda_3(Syn)$.
  - `trainer.py` - Manages the phases of cognitive training (Redundant -> Pruning -> Specialization).
- `src/sipnet/application/execution/`
  - `forward_pass.py` - Handles routing from encoding to buffering, integration, and output.

#### Presentation/Interface Layer
- `src/sipnet/interfaces/cli/` - Command-line tools for running tests or training.
- `src/sipnet/interfaces/api/` - Future REST interfaces if necessary.

## Verification Plan

### Automated Tests
- Unit testing each IPP module (`StorageNode`, `TransferBus`, `SynergyHub`) isolated from the graph.
- Mathematical validation tests for the information-theoretic estimators.
- Integration test for the composite loss function with dummy data.

### Manual Verification
- Track representational complexity outputs during training.
- Extract TE/AIS/PID logs on a small NLP test case to verify extreme interpretability (i.e. verifying that `StorageNode` buffers context actively, and `TransferBus` routes it dynamically based on TE).

# Integrations

## Internal Integrations
- **SIPNet Core**: Integration between `domain.nodes` (StorageNode, SynergyHub, TransferBus) and `domain.network` (SIPNet, SIPLayer).
- **Estimation Engine**: Integration between `infrastructure.information_theory` estimators and the training/execution layers.
- **Application Layer**: Integration between `application.training.trainer` and `application.execution.nlp_generators`.

## External Integrations
- **PyTorch**: Used for the neural network backend, backpropagation, and tensor operations.
- **Scientific Stack**: NumPy and SciPy used for information-theoretic calculations and linear algebra.
- **CLI/Interfaces**: Basic CLI integration in `interfaces.cli`.

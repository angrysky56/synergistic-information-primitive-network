# Project Structure

## Directory Map
```
src/sipnet/
├── application/           # High-level application logic
│   ├── execution/         # Real-world usage (e.g., NLP generation)
│   └── training/          # Training loops and loss functions
├── domain/                # Core business logic and models
│   ├── network/           # SIPNet graph and layer definitions
│   └── nodes/             # Primitive node types (Storage, Synergy, Bus)
├── infrastructure/        # Low-level technical implementations
│   └── information_theory/# PID, AIS, and TE estimators
└── interfaces/            # Entry points for users/systems
    ├── api/               # API definitions (placeholder)
    └── cli/               # CLI tools
```

## Key Files
- `src/sipnet/domain/network/graph.py`: The main `SIPNet` model class.
- `src/sipnet/domain/network/sip_layer.py`: The `SIPLayer` implementation.
- `src/sipnet/domain/nodes/storage_node.py`: State management node.
- `src/sipnet/domain/nodes/synergy_hub.py`: Synergistic interaction hub.
- `src/sipnet/infrastructure/information_theory/pid_estimator.py`: PID estimation logic.
- `src/sipnet/application/training/trainer.py`: Model training logic.

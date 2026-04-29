# Coding Conventions

## Language & Formatting
- **Language**: Python 3.13+
- **Formatting**: Handled by Ruff with a line length of 100.
- **Imports**: Alphabetical order within groups (stdlib, third-party, local). Handled by Ruff's `I` (isort) rules.

## Type Safety
- **Strict Typing**: All functions and methods MUST have type hints for all arguments and return values.
- **Mypy**: The project uses `mypy` in strict mode. `disallow_untyped_defs = true` is enforced.

## Documentation
- **Docstrings**: All classes and public methods must have docstrings following the Google/Sphinx style (Args, Returns).
- **Comments**: Use comments to explain complex information-theoretic logic or neural network architectures.

## Architecture Patterns
- **Module Structure**: Follow the Domain-Driven Design (DDD) layers: Application, Domain, Infrastructure, and Interfaces.
- **PyTorch**: Subclass `nn.Module` for all neural components. Implement `forward` and `reset_memory`/`reset_state` where applicable.

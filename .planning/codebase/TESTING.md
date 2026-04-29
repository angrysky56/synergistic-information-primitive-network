# Testing Standards

## Framework
- **Pytest**: Used for all levels of testing.

## Test Structure
- **Unit Tests**: Located in `tests/unit/`. Mirrored directory structure of `src/`.
- **Integration Tests**: Located in `tests/integration/`. Tests interactions between layers (e.g., SIPLayer integration).
- **Smoke Tests**: Basic execution tests in scripts (e.g., `test_subclass.py`).

## Test Patterns
- **Mocking**: Use `unittest.mock` or pytest fixtures where necessary, though many tests use real (randomized) tensors.
- **Verification**: Verify output shapes, state persistence, and mathematical consistency (e.g., non-zero activity).
- **Coverage**: Aim for high coverage in `domain` and `infrastructure` layers.

## Running Tests
```bash
# Run all tests
pytest

# Run tests for a specific module
pytest tests/unit/domain/test_nodes.py

# Run with coverage
pytest --cov=src
```

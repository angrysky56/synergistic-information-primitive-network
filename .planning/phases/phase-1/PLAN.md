# Phase 1: Diagnostic Audit & Typing Hardening

## Objective
Achieve strict typing and resolve structural bug noise to establish a stable foundation for the SIP-Net architecture.

## Tasks

### 1. Project Infrastructure Fixes
- [ ] **Task 1.1**: Update `pyproject.toml` to include `src` in `pytest` pythonpath.
- [ ] **Task 1.2**: Create a `types.py` module in `domain/common` to define `TypedDict` for layer outputs and training metrics.
- [ ] **Task 1.3**: Refactor `CompositeLoss.forward` and `SIPLayer.forward_step` to use the new `TypedDict` for better typing.

### 2. Domain Model Hardening
- [ ] **Task 2.1**: Audit `StorageNode`, `TransferBus`, and `SynergyHub` for proper `nn.Module` subclassing and `forward` signature compliance.
- [ ] **Task 2.2**: Fix `StorageNode.state` buffer reassignment issue (use `copy_` or proper buffer management).
- [ ] **Task 2.3**: Resolve `linalg.eigvals` vs `torch.linalg.eigvals` confusion in `StorageNode`.

### 3. Application & Script Typing
- [ ] **Task 3.1**: Add missing type annotations to all scripts in `scripts/`.
- [ ] **Task 3.2**: Fix `CognitiveTrainer.train_epoch` typing for `DataLoader` and `loss_dict`.
- [ ] **Task 3.3**: Ensure `CompositeLoss` does not use `type: ignore` if possible.

### 4. Verification Loop
- [ ] **Task 4.1**: Run `uv run mypy src scripts` and ensure 0 errors.
- [ ] **Task 4.2**: Run `PYTHONPATH=src uv run pytest` and ensure all tests pass.

## Success Criteria
- [ ] `mypy` returns 0 errors on `src` and `scripts` directories.
- [ ] `pytest` successfully collects and runs all tests in `tests/` without `ModuleNotFoundError`.
- [ ] All `nn.Module` subclasses have clean type hints and follow PyTorch best practices.

# Phase 1 Research: Diagnostic Audit

## Mypy Audit Results
- **Total Errors**: 39
- **Key Categories**:
    1. **Invalid Indexing**: `loss_function.py` uses string keys on what mypy thinks are Tensors (due to broad `dict[str, torch.Tensor]` type hint).
    2. **Missing Type Args**: `DataLoader` needs generic type parameters.
    3. **Untyped Calls**: `backward()` and `main()` calls in scripts.
    4. **Missing Annotations**: Scripts lack return types and arg types.

## Pytest Audit Results
- **Status**: FAILED (Collection Error)
- **Error**: `ModuleNotFoundError: No module named 'sipnet'`
- **Cause**: `src` directory is not in the pythonpath during test collection.
- **Fix**: Add `pythonpath = ["src"]` to `pyproject.toml` or run with `PYTHONPATH=src`.

## Structural Observations
- **StorageNode**: Uses `linalg.eigvals` (imported from `torch`). Should use `torch.linalg.eigvals`.
- **Buffer Management**: `StorageNode.state` is reassigned in `forward`, which might break buffer tracking if not careful.
- **Layer Output Typing**: The dictionary returned by `SIPLayer.forward_step` is complex and untyped, leading to downstream typing errors in the loss function.

import torch
from .kernel_utils import compute_kernel, compute_h2_from_kernel

def estimate_te(
    source_past: torch.Tensor,
    target_present: torch.Tensor,
    target_past: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Estimates Transfer Entropy: TE_{Y->X} = I(X_present; Y_past | X_past).
    Using Shannon's identity for conditional mutual information:
    I(A; B | C) = H(A, C) + H(B, C) - H(A, B, C) - H(C)

    In Rényi's H_2 space via kernel methods:
    TE = H_2(Tp, Tpast) + H_2(Spast, Tpast) - H_2(Tp, Spast, Tpast) - H_2(Tpast)

    Args:
        source_past: Tensor of shape [batch_size, dim_s] representing source at t-1.
        target_present: Tensor of shape [batch_size, dim_t] representing target at t.
        target_past: Tensor of shape [batch_size, dim_t] representing target at t-1.

    Returns:
        TE estimate (scalar tensor).
    """
    K_Tp = compute_kernel(target_present)
    K_Tpast = compute_kernel(target_past)
    K_Spast = compute_kernel(source_past)

    # Joint Spaces via Hadamard Products
    K_Tp_Tpast = K_Tp * K_Tpast
    K_Spast_Tpast = K_Spast * K_Tpast
    K_Tp_Spast_Tpast = K_Tp * K_Spast * K_Tpast

    # Entropies
    H2_Tp_Tpast = compute_h2_from_kernel(K_Tp_Tpast, eps)
    H2_Spast_Tpast = compute_h2_from_kernel(K_Spast_Tpast, eps)
    H2_Tp_Spast_Tpast = compute_h2_from_kernel(K_Tp_Spast_Tpast, eps)
    H2_Tpast = compute_h2_from_kernel(K_Tpast, eps)

    # Transfer Entropy
    te = H2_Tp_Tpast + H2_Spast_Tpast - H2_Tp_Spast_Tpast - H2_Tpast

    return torch.clamp(te, min=0.0)

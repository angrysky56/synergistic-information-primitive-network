import torch
from .kernel_utils import compute_kernel, compute_h2_from_kernel

def estimate_pid(
    s1: torch.Tensor,
    s2: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8
) -> dict[str, torch.Tensor]:
    """
    Partial Information Decomposition (PID) using Minimum Mutual Information (MMI)
    in a Matrix-Rényi's H_2 space.

    Decomposes the total joint mutual information I(Target; S1, S2) into:
    - Synergy
    - Redundancy
    - Unique 1
    - Unique 2

    Args:
        s1: Source 1 (e.g. FF Signal) [batch_size, dim]
        s2: Source 2 (e.g. Context Signal) [batch_size, dim]
        target: Target space [batch_size, dim]

    Returns:
        Dictionary of PID components.
    """
    K_T = compute_kernel(target)
    K_S1 = compute_kernel(s1)
    K_S2 = compute_kernel(s2)
    K_S1_S2 = K_S1 * K_S2

    # Individual Entropies
    H2_T = compute_h2_from_kernel(K_T, eps)
    H2_S1 = compute_h2_from_kernel(K_S1, eps)
    H2_S2 = compute_h2_from_kernel(K_S2, eps)
    H2_S1_S2 = compute_h2_from_kernel(K_S1_S2, eps)

    # Joint Spaces
    H2_T_S1 = compute_h2_from_kernel(K_T * K_S1, eps)
    H2_T_S2 = compute_h2_from_kernel(K_T * K_S2, eps)
    H2_T_S1_S2 = compute_h2_from_kernel(K_T * K_S1_S2, eps)

    # Mutual Informations I_2(X;Y) = H_2(X) + H_2(Y) - H_2(X,Y)
    I2_T_S1 = torch.clamp(H2_T + H2_S1 - H2_T_S1, min=0.0)
    I2_T_S2 = torch.clamp(H2_T + H2_S2 - H2_T_S2, min=0.0)
    I2_T_S1_S2 = torch.clamp(H2_T + H2_S1_S2 - H2_T_S1_S2, min=0.0)

    # Redundancy via MMI
    redundancy = torch.min(I2_T_S1, I2_T_S2)

    # Synergy via PID conservation:
    # I(T; S1, S2) = Synergy + Unique_1 + Unique_2 + Redundancy
    # and Unique_1 = I(T; S1) - Redundancy
    # => Synergy = I(T; S1, S2) - I(T; S1) - I(T; S2) + Redundancy
    synergy = I2_T_S1_S2 - I2_T_S1 - I2_T_S2 + redundancy
    synergy = torch.clamp(synergy, min=0.0)

    unique_1 = torch.clamp(I2_T_S1 - redundancy, min=0.0)
    unique_2 = torch.clamp(I2_T_S2 - redundancy, min=0.0)

    return {
        "synergy": synergy,
        "redundancy": redundancy,
        "unique_1": unique_1,
        "unique_2": unique_2
    }

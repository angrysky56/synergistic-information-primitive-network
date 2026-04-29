"""
Estimation of the Analogue Integrated Signal (AIS) and related information-theoretic quantities.
"""

import torch

from .kernel_utils import (
    compute_normalized_kernel,
    joint_renyi_entropy_2,
    renyi_entropy,
    renyi_entropy_2,
)


def estimate_mutual_information_renyi(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 2.0
) -> torch.Tensor:
    """I_alpha(X; Y) = H_alpha(X) + H_alpha(Y) - H_alpha(X, Y)"""
    K_x = compute_normalized_kernel(x)
    K_y = compute_normalized_kernel(y)

    h_x = renyi_entropy(K_x, alpha=alpha)
    h_y = renyi_entropy(K_y, alpha=alpha)

    # For joint entropy with alpha != 2, we need a generalized joint kernel function
    # or just use the Hadamard product and re-normalize.
    # Currently joint_renyi_entropy_2 is specific to alpha=2 in its name but logic is same.
    # Let's generalize it.

    joint_K = K_x * K_y
    trace = torch.trace(joint_K)
    if trace < 1e-12:
        h_xy = torch.log2(torch.tensor(float(K_x.shape[0]), device=K_x.device))
    else:
        h_xy = renyi_entropy(joint_K / trace, alpha=alpha)

    mi = h_x + h_y - h_xy
    return torch.clamp(mi, min=0.0)


def estimate_ais(
    past: torch.Tensor, present: torch.Tensor, alpha: float = 2.0
) -> torch.Tensor:
    return estimate_mutual_information_renyi(past, present, alpha=alpha)

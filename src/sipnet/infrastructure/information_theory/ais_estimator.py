import torch
from .kernel_utils import estimate_mutual_information_renyi

def estimate_ais(
    past: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    """
    Estimates Active Information Storage: I_2(X_past; X_present).
    Uses Matrix-based Rényi's alpha=2 Entropy via kernel methods.

    Args:
        past: Tensor of shape [batch_size, dim] representing state at t-1.
        present: Tensor of shape [batch_size, dim] representing state at t.

    Returns:
        AIS estimate.
    """
    return estimate_mutual_information_renyi(past, present)

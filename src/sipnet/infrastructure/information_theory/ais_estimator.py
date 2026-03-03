import torch
from .kernel_utils import compute_normalized_kernel, renyi_entropy_2, joint_renyi_entropy_2

def estimate_mutual_information_renyi(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """I_2(X; Y) = H_2(X) + H_2(Y) - H_2(X, Y)"""
    K_x = compute_normalized_kernel(x)
    K_y = compute_normalized_kernel(y)

    h_x = renyi_entropy_2(K_x)
    h_y = renyi_entropy_2(K_y)
    h_xy = joint_renyi_entropy_2(K_x, K_y)

    mi = h_x + h_y - h_xy
    return torch.clamp(mi, min=0.0)

def estimate_ais(past: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    return estimate_mutual_information_renyi(past, present)

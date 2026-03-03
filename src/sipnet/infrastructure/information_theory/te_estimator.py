import torch
from .kernel_utils import compute_normalized_kernel, renyi_entropy_2, joint_renyi_entropy_2

def estimate_te(
    source_past: torch.Tensor,
    target_present: torch.Tensor,
    target_past: torch.Tensor
) -> torch.Tensor:
    """
    TE = I(Tp; Sp | Tpast) = H(Tp, Tpast) + H(Sp, Tpast) - H(Tp, Sp, Tpast) - H(Tpast)
    """
    K_tp = compute_normalized_kernel(target_present)
    K_sp = compute_normalized_kernel(source_past)
    K_t_past = compute_normalized_kernel(target_past)

    h_tp_tpast = joint_renyi_entropy_2(K_tp, K_t_past)
    h_sp_tpast = joint_renyi_entropy_2(K_sp, K_t_past)
    h_tp_sp_tpast = joint_renyi_entropy_2(K_tp, K_sp, K_t_past)
    h_tpast = renyi_entropy_2(K_t_past)

    te = h_tp_tpast + h_sp_tpast - h_tp_sp_tpast - h_tpast
    return torch.clamp(te, min=0.0)

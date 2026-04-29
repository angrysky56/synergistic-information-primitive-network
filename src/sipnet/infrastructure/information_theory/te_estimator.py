import torch

from .kernel_utils import (
    compute_normalized_kernel,
    renyi_entropy,
)


def estimate_te(
    source_past: torch.Tensor,
    target_present: torch.Tensor,
    target_past: torch.Tensor,
    alpha: float = 2.0,
) -> torch.Tensor:
    """
    TE = I(Tp; Sp | Tpast) = H(Tp, Tpast) + H(Sp, Tpast) - H(Tp, Sp, Tpast) - H(Tpast)
    """
    K_tp = compute_normalized_kernel(target_present)
    K_sp = compute_normalized_kernel(source_past)
    K_t_past = compute_normalized_kernel(target_past)

    def get_joint_entropy(*ks: torch.Tensor) -> torch.Tensor:
        joint_K = ks[0]
        for k in ks[1:]:
            joint_K = joint_K * k
        trace = torch.trace(joint_K)
        if trace < 1e-12:
            return torch.log2(torch.tensor(float(ks[0].shape[0]), device=ks[0].device))
        return renyi_entropy(joint_K / trace, alpha=alpha)

    h_tp_tpast = get_joint_entropy(K_tp, K_t_past)
    h_sp_tpast = get_joint_entropy(K_sp, K_t_past)
    h_tp_sp_tpast = get_joint_entropy(K_tp, K_sp, K_t_past)
    h_tpast = renyi_entropy(K_t_past, alpha=alpha)

    te = h_tp_tpast + h_sp_tpast - h_tp_sp_tpast - h_tpast
    return torch.clamp(te, min=0.0)

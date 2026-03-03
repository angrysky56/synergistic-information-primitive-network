import torch
from .ais_estimator import estimate_mutual_information_renyi

def estimate_pid_renyi(
    s1: torch.Tensor,
    s2: torch.Tensor,
    target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Decomposes total MI into Redundancy, Unique, and Synergy using MMI.
    """
    mi_s1 = estimate_mutual_information_renyi(s1, target)
    mi_s2 = estimate_mutual_information_renyi(s2, target)

    # For joint MI, we concatenate features (or use joint kernels, concatenation is simpler here)
    sources_combined = torch.cat([s1, s2], dim=1)
    mi_total = estimate_mutual_information_renyi(sources_combined, target)

    # MMI Approximation for Redundancy
    redundancy = torch.min(mi_s1, mi_s2)

    # Synergy = I(T; S1, S2) - I(T; S1) - I(T; S2) + Redundancy
    synergy = mi_total - mi_s1 - mi_s2 + redundancy

    unique1 = mi_s1 - redundancy
    unique2 = mi_s2 - redundancy

    return {
        'redundancy': redundancy,
        'unique1': torch.clamp(unique1, min=0.0),
        'unique2': torch.clamp(unique2, min=0.0),
        'synergy': torch.clamp(synergy, min=0.0)
    }

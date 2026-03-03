import torch

def compute_normalized_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized Gaussian RBF kernel matrix A_X = K_X / Tr(K_X).
    Uses a median-distance heuristic for the bandwidth sigma.
    """
    batch_size = x.shape[0]
    if batch_size < 2:
        return torch.eye(batch_size, device=x.device)

    # Pairwise squared distances: ||x_i - x_j||^2
    dist_sq = torch.cdist(x, x, p=2) ** 2

    # Median heuristic for sigma
    triu_idx = torch.triu_indices(batch_size, batch_size, offset=1)
    if triu_idx.shape[1] > 0:
        med_dist_sq = torch.median(dist_sq[triu_idx[0], triu_idx[1]])
        sigma_sq = (med_dist_sq / 2.0).clamp(min=1e-4)
    else:
        sigma_sq = torch.tensor(1.0, device=x.device)

    # Compute RBF Kernel
    K = torch.exp(-dist_sq / (2 * sigma_sq))

    # Normalize by trace so eigenvalues sum to 1
    trace = torch.trace(K)
    return K / trace.clamp(min=1e-6)

def renyi_entropy_2(K: torch.Tensor) -> torch.Tensor:
    """
    Computes Rényi's alpha=2 Collision Entropy directly from the kernel.
    H_2(X) = -log2(Tr(A_X^2)) = -log2(sum(A_X(i,j)^2))
    """
    # Trace of squared matrix is sum of squared elements
    tr_k_sq = torch.sum(K ** 2)
    return -torch.log2(tr_k_sq.clamp(min=1e-8))

def joint_renyi_entropy_2(*kernels: torch.Tensor) -> torch.Tensor:
    """
    Computes joint entropy using the Hadamard (element-wise) product of kernels.
    """
    joint_K = kernels[0]
    for K in kernels[1:]:
        joint_K = joint_K * K # Hadamard product

    # Re-normalize the joint kernel
    trace = torch.trace(joint_K)
    joint_K_norm = joint_K / trace.clamp(min=1e-6)

    return renyi_entropy_2(joint_K_norm)

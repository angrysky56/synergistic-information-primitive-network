from typing import cast

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

    # Ensure no NaNs from exp (though rare with positive dist_sq)
    if torch.any(torch.isnan(K)):
        K = torch.nan_to_num(K, nan=0.0)

    # Normalize by trace so eigenvalues sum to 1
    trace = torch.trace(K)
    if trace < 1e-12:
        # Fallback to identity if trace is nearly zero
        return torch.eye(batch_size, device=x.device)

    return K / trace


def renyi_entropy_2(K: torch.Tensor) -> torch.Tensor:
    """
    Computes Rényi's alpha=2 Collision Entropy directly from the kernel.
    H_2(X) = -log2(Tr(A_X^2)) = -log2(sum(A_X(i,j)^2))
    """
    # Trace of squared matrix is sum of squared elements
    tr_k_sq = torch.sum(K**2)
    return -torch.log2(tr_k_sq.clamp(min=1e-8))


def joint_renyi_entropy_2(*kernels: torch.Tensor) -> torch.Tensor:
    """
    Computes joint entropy using the Hadamard (element-wise) product of kernels.
    """
    joint_K = kernels[0]
    for K in kernels[1:]:
        joint_K = joint_K * K  # Hadamard product

    # Re-normalize the joint kernel
    trace = torch.trace(joint_K)
    if trace < 1e-12:
        # If joint kernel is zero, variables are effectively independent or disjoint
        # Return a high but finite entropy (or sum of entropies?)
        # For now, return renyi_entropy_2 of identity (max entropy)
        return torch.log2(torch.tensor(float(joint_K.shape[0]), device=joint_K.device))

    joint_K_norm = joint_K / trace
    return renyi_entropy_2(joint_K_norm)


def renyi_entropy(
    K: torch.Tensor, alpha: float = 2.0, approx: bool = False
) -> torch.Tensor:
    """
    Computes Matrix-Rényi alpha-entropy.
    If alpha=2, uses the efficient Tr(A^2) implementation.
    Otherwise, uses eigenvalue decomposition (O(N^3)) or approximation (O(N^2)).
    """
    if abs(alpha - 2.0) < 1e-6:
        return renyi_entropy_2(K)

    if abs(alpha - 1.0) < 1e-6:
        if approx:
            return von_neumann_entropy_approx(K)
        return von_neumann_entropy(K)

    if approx:
        return renyi_entropy_approx(K, alpha=alpha)

    # Eigenvalues for generalized alpha
    lambdas = torch.linalg.eigvalsh(K)
    lambdas = torch.clamp(lambdas, min=1e-10)
    sum_lambdas_alpha = torch.sum(lambdas**alpha)

    entropy = (1.0 / (1.0 - alpha)) * torch.log2(sum_lambdas_alpha.clamp(min=1e-12))
    return torch.clamp(entropy, min=0.0)


def renyi_entropy_approx(
    K: torch.Tensor, alpha: float = 2.0, degree: int = 30, num_vectors: int = 50
) -> torch.Tensor:
    """
    Approximates Matrix-Rényi alpha-entropy in O(N^2) using
    Chebyshev polynomial expansion and Randomized Trace estimation.
    """
    if abs(alpha - 2.0) < 1e-6:
        return renyi_entropy_2(K)

    # 1. Compute Chebyshev coefficients for f(x) = x^alpha on [0, 1]
    # x in [0, 1] maps to t in [-1, 1] via t = 2x - 1
    # We approximate g(t) = ((t+1)/2)^alpha
    import numpy as np

    def g(t: float | np.ndarray) -> float | np.ndarray:
        return ((t + 1) / 2) ** alpha

    # Nodes for Chebyshev approximation
    nodes = np.cos(np.pi * (np.arange(degree) + 0.5) / degree)
    values = g(nodes)
    coeffs = np.zeros(degree)
    for k in range(degree):
        coeffs[k] = (2.0 / degree) * np.sum(
            values * np.cos(np.pi * k * (np.arange(degree) + 0.5) / degree)
        )
    # coeffs[0] needs to be halved for the standard sum
    coeffs[0] /= 2.0

    coeffs_torch = torch.tensor(coeffs, device=K.device, dtype=K.dtype)

    # 2. Randomized Trace Estimation: Tr(f(A)) = E[z^T f(A) z]
    # z is a Rademacher vector (elements are +/-1)
    batch_size = K.shape[0]
    z = torch.randint(0, 2, (batch_size, num_vectors), device=K.device).float() * 2 - 1

    # Map A to [-1, 1] domain: T = 2A - I
    # Since K is normalized, its eigenvalues are in [0, 1].
    I = torch.eye(batch_size, device=K.device)
    T_mat = 2 * K - I

    # Recursive computation of Tr(T_k(T_mat))
    # v_k = T_k(T_mat) z
    # v_0 = z
    # v_1 = T_mat @ z
    # v_{k+1} = 2 * T_mat @ v_k - v_{k-1}

    v_prev = z
    v_curr = torch.matmul(T_mat, z)

    # tr_f_A = coeffs[0] * z^T v_0 + coeffs[1] * z^T v_1 + ...
    tr_f_A = coeffs_torch[0] * torch.sum(z * v_prev) / num_vectors
    tr_f_A += coeffs_torch[1] * torch.sum(z * v_curr) / num_vectors

    for k in range(2, degree):
        v_next = 2 * torch.matmul(T_mat, v_curr) - v_prev
        tr_f_A += coeffs_torch[k] * torch.sum(z * v_next) / num_vectors
        v_prev = v_curr
        v_curr = v_next

    entropy = (1.0 / (1.0 - alpha)) * torch.log2(tr_f_A.clamp(min=1e-12))
    return torch.clamp(entropy, min=0.0)


def von_neumann_entropy(K: torch.Tensor) -> torch.Tensor:
    """
    Computes von Neumann entropy: H_1(A) = -Tr(A log2 A) = -sum(lambda_i * log2 lambda_i)
    """
    lambdas = torch.linalg.eigvalsh(K)
    lambdas = torch.clamp(lambdas, min=1e-10)

    entropy = -torch.sum(lambdas * torch.log2(lambdas))
    return torch.clamp(entropy, min=0.0)


def von_neumann_entropy_approx(
    K: torch.Tensor, degree: int = 30, num_vectors: int = 50
) -> torch.Tensor:
    """
    Approximates von Neumann entropy in O(N^2) using Chebyshev
    polynomial expansion of f(x) = -x * log2(x).
    """
    import numpy as np

    def g(t: float | np.ndarray) -> float | np.ndarray:
        x = (t + 1) / 2
        # Use a small epsilon for log stability at x=0
        x = np.clip(x, 1e-12, 1.0)
        return cast(float | np.ndarray, -x * np.log2(x))

    # Nodes for Chebyshev approximation
    nodes = np.cos(np.pi * (np.arange(degree) + 0.5) / degree)
    values = g(nodes)
    coeffs = np.zeros(degree)
    for k in range(degree):
        coeffs[k] = (2.0 / degree) * np.sum(
            values * np.cos(np.pi * k * (np.arange(degree) + 0.5) / degree)
        )
    coeffs[0] /= 2.0

    coeffs_torch = torch.tensor(coeffs, device=K.device, dtype=K.dtype)

    batch_size = K.shape[0]
    z = torch.randint(0, 2, (batch_size, num_vectors), device=K.device).float() * 2 - 1
    I = torch.eye(batch_size, device=K.device)
    T_mat = 2 * K - I

    v_prev = z
    v_curr = torch.matmul(T_mat, z)

    tr_f_A = coeffs_torch[0] * torch.sum(z * v_prev) / num_vectors
    tr_f_A += coeffs_torch[1] * torch.sum(z * v_curr) / num_vectors

    for k in range(2, degree):
        v_next = 2 * torch.matmul(T_mat, v_curr) - v_prev
        tr_f_A += coeffs_torch[k] * torch.sum(z * v_next) / num_vectors
        v_prev = v_curr
        v_curr = v_next

    return torch.clamp(tr_f_A, min=0.0)

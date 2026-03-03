import torch

def compute_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gaussian RBF kernel matrix K_X for a given batch using Scott's/Silverman's rule of thumb.
    """
    batch_size, dim = x.shape

    # Handle edge case where batch_size is small or dimensions are 0
    if batch_size < 2 or dim == 0:
        return torch.eye(batch_size, device=x.device)

    # Compute squared pairwise distances (N x N)
    dist_sq = torch.cdist(x, x, p=2) ** 2

    # Use the median heuristic for robust bandwidth estimation in high dimensions
    # 2 * sigma^2 = median(||x_i - x_j||^2)
    median_sq = torch.median(dist_sq)
    if median_sq == 0.0:
        median_sq = torch.tensor(1e-8, device=x.device)

    # Compute the RBF kernel K_X
    # K(i,j) = exp( - ||x_i - x_j||^2 / median_sq )
    kappa = torch.exp(-dist_sq / median_sq)

    return kappa

def compute_h2_from_kernel(K: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes Rényi's Entropy of order 2 from a kernel matrix.
    H_2(X) = -log2( Tr(A_X^2) )  where A_X = K_X / Tr(K_X)
    Since K_X is an RBF kernel, K(i,i) = 1, so Tr(K_X) = N.
    """
    N = K.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=K.device, requires_grad=True)

    # Normalize Kernel to create Gram probability mass matrix
    A = K / N

    # The trace of A^2 is equivalent to the sum of all squared elements of A
    tr_A2 = torch.sum(A ** 2)

    # Compute H_2
    h2 = -torch.log2(torch.clamp(tr_A2, min=eps))
    return h2

def estimate_mutual_information_renyi(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Estimates Mutual Information I_2(X; Y) = H_2(X) + H_2(Y) - H_2(X, Y)
    """
    K_x = compute_kernel(x)
    K_y = compute_kernel(y)

    # Joint kernel via Schur Product Theorem
    K_xy = K_x * K_y

    H2_x = compute_h2_from_kernel(K_x, eps)
    H2_y = compute_h2_from_kernel(K_y, eps)
    H2_xy = compute_h2_from_kernel(K_xy, eps)

    mi = H2_x + H2_y - H2_xy
    return torch.clamp(mi, min=0.0)

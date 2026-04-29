import torch
from sipnet.infrastructure.information_theory.kernel_utils import compute_normalized_kernel, renyi_entropy_2

def test_singularities():
    # 1. Identical samples
    x_ident = torch.ones(10, 5)
    K_ident = compute_normalized_kernel(x_ident)
    h_ident = renyi_entropy_2(K_ident)
    print(f"Entropy of identical samples: {h_ident.item()}")
    
    # 2. All zeros
    x_zeros = torch.zeros(10, 5)
    K_zeros = compute_normalized_kernel(x_zeros)
    h_zeros = renyi_entropy_2(K_zeros)
    print(f"Entropy of zero samples: {h_zeros.item()}")
    
    # 3. Single sample
    x_single = torch.randn(1, 5)
    K_single = compute_normalized_kernel(x_single)
    h_single = renyi_entropy_2(K_single)
    print(f"Entropy of single sample: {h_single.item()}")
    
    # 4. Very large values (potential underflow in RBF)
    x_large = torch.randn(10, 5) * 1e6
    K_large = compute_normalized_kernel(x_large)
    h_large = renyi_entropy_2(K_large)
    print(f"Entropy of large value samples: {h_large.item()}")

if __name__ == "__main__":
    test_singularities()

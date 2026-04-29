import torch
from sipnet.infrastructure.information_theory.kernel_utils import compute_normalized_kernel, renyi_entropy

def test_batch_scaling():
    print(f"{'N':<10} | {'Entropy (Exact)':<20} | {'Entropy (Approx)':<20}")
    print("-" * 55)
    
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    dim = 16
    
    for N in batch_sizes:
        x = torch.randn(N, dim)
        K = compute_normalized_kernel(x)
        h_exact = renyi_entropy(K, alpha=1.5, approx=False)
        h_approx = renyi_entropy(K, alpha=1.5, approx=True)
        print(f"{N:<10} | {h_exact.item():<20.4f} | {h_approx.item():<20.4f}")

if __name__ == "__main__":
    test_batch_scaling()

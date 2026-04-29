import torch
import time
from sipnet.infrastructure.information_theory.kernel_utils import compute_normalized_kernel, renyi_entropy

def cross_validate_estimators():
    print(f"{'Alpha':<10} | {'Exact':<10} | {'Approx':<10} | {'Error (%)':<10} | {'Exact Time':<10} | {'Approx Time':<10}")
    print("-" * 75)
    
    batch_size = 512
    dim = 64
    x = torch.randn(batch_size, dim)
    K = compute_normalized_kernel(x)
    
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    for alpha in alphas:
        # Exact
        start = time.time()
        h_exact = renyi_entropy(K, alpha=alpha, approx=False)
        exact_time = time.time() - start
        
        # Approx
        start = time.time()
        h_approx = renyi_entropy(K, alpha=alpha, approx=True)
        approx_time = time.time() - start
        
        error = abs(h_exact - h_approx) / (h_exact + 1e-9) * 100
        
        print(f"{alpha:<10.2f} | {h_exact.item():<10.4f} | {h_approx.item():<10.4f} | {error.item():<10.2f}% | {exact_time:<10.4f} | {approx_time:<10.4f}")

if __name__ == "__main__":
    cross_validate_estimators()

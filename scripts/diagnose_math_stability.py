import matplotlib.pyplot as plt
import numpy as np
import torch

from sipnet.infrastructure.information_theory.kernel_utils import (
    compute_normalized_kernel,
    renyi_entropy_2,
)


def diagnose_stability() -> None:
    print("Diagnosing Matrix-Rényi Stability...")

    # 1. Variance Sensitivity
    variances = torch.logspace(-6, 2, steps=20)
    entropies = []

    for v in variances:
        # Create data with small variance
        x = torch.randn(100, 10) * v
        K = compute_normalized_kernel(x)
        h = renyi_entropy_2(K)
        entropies.append(h.item())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.semilogx(variances.numpy(), entropies)
    plt.title("Entropy vs Variance")
    plt.xlabel("Variance")
    plt.ylabel("H_2")

    # 2. Correlation Sensitivity
    correlations = torch.linspace(0, 0.999, steps=20)
    entropies_corr = []

    for c in correlations:
        # Create highly correlated data
        base = torch.randn(100, 10)
        noise = torch.randn(100, 10)
        x = base * c + noise * (1 - c)
        K = compute_normalized_kernel(x)
        h = renyi_entropy_2(K)
        entropies_corr.append(h.item())

    plt.subplot(1, 2, 2)
    plt.plot(correlations.numpy(), entropies_corr)
    plt.title("Entropy vs Correlation")
    plt.xlabel("Correlation Factor")
    plt.ylabel("H_2")

    plt.tight_layout()
    plt.savefig("stability_diagnostic.png")
    print("Saved stability_diagnostic.png")


if __name__ == "__main__":
    diagnose_stability()

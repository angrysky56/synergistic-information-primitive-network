import torch

from sipnet.infrastructure.information_theory.kernel_utils import (
    compute_normalized_kernel,
    renyi_entropy,
)


def test_gradient_flow() -> None:
    print("Testing gradient flow for Rényi entropy estimators...")
    batch_size = 32
    dim = 16
    x = torch.randn(batch_size, dim, requires_grad=True)

    alphas = [0.5, 1.0, 1.5, 2.0, 3.0]

    for alpha in alphas:
        # Exact
        K = compute_normalized_kernel(x)
        h = renyi_entropy(K, alpha=alpha, approx=False)
        h.backward()  # type: ignore[no-untyped-call]

        assert x.grad is not None
        if torch.isnan(x.grad).any():
            print(f"FAILED: Exact Alpha={alpha} produced NaNs in gradients.")
        else:
            print(f"PASSED: Exact Alpha={alpha} gradient is stable.")

        x.grad.zero_()

        # Approx
        K = compute_normalized_kernel(x)
        h_approx = renyi_entropy(K, alpha=alpha, approx=True)
        h_approx.backward()  # type: ignore[no-untyped-call]

        assert x.grad is not None
        if torch.isnan(x.grad).any():
            print(f"FAILED: Approx Alpha={alpha} produced NaNs in gradients.")
        else:
            print(f"PASSED: Approx Alpha={alpha} gradient is stable.")

        x.grad.zero_()


if __name__ == "__main__":
    test_gradient_flow()

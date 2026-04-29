import torch

from sipnet.infrastructure.information_theory.kernel_utils import (
    compute_normalized_kernel,
    renyi_entropy_2,
)


def diagnose_gradients():
    print("Diagnosing Gradient Stability...")

    # 1. Vanishing Gradients at low variance
    variances = torch.logspace(-8, 1, steps=10)

    for v in variances:
        x = torch.randn(10, 5) * v
        x.requires_grad_(True)
        K = compute_normalized_kernel(x)
        h = renyi_entropy_2(K)
        h.backward()

        grad_norm = x.grad.norm().item()
        print(f"Var: {v.item():.1e} | H: {h.item():.4f} | Grad Norm: {grad_norm:.4e}")

    # 2. Exploding Gradients at high similarity
    print("\nDiagnosing near-identity correlation...")
    base = torch.randn(10, 5)
    epsilons = torch.logspace(-8, -1, steps=10)

    for eps in epsilons:
        x = (base + torch.randn(10, 5) * eps).clone().detach().requires_grad_(True)
        K = compute_normalized_kernel(x)
        h = renyi_entropy_2(K)
        h.backward()

        grad_norm = x.grad.norm().item()
        print(f"Eps: {eps.item():.1e} | H: {h.item():.4f} | Grad Norm: {grad_norm:.4e}")


def simulate_phase_transitions():
    print("\nDiagnosing Multi-Phase Transition Stability...")
    from sipnet.application.training.loss_function import CompositeLoss
    import torch.nn as nn

    # Dummy model and optimizer
    model = nn.Sequential(nn.Linear(10, 10))
    loss_fn = CompositeLoss(nn.MSELoss())
    
    # Mock StepOutput
    def get_mock_output(batch_size=10):
        return {
            "logits": torch.randn(batch_size, 10, requires_grad=True),
            "layer_outputs": [{
                "context_state": torch.randn(batch_size, 10, requires_grad=True),
                "final_rep": torch.randn(batch_size, 10, requires_grad=True),
                "agg_ff_signal": torch.randn(batch_size, 10, requires_grad=True),
                "agg_ctx_signal": torch.randn(batch_size, 10, requires_grad=True),
                "ctx_signals": [torch.randn(batch_size, 10, requires_grad=True)]
            }],
            "prev_layer_outputs": [{
                "context_state": torch.randn(batch_size, 10, requires_grad=True),
                "final_rep": torch.randn(batch_size, 10, requires_grad=True)
            }]
        }

    targets = torch.randn(10, 10)
    
    phases = [
        {"ais": 0.1, "te": 0.1, "synergy": 0.1},  # Phase 1
        {"ais": 0.1, "te": 1.0, "synergy": 0.1},  # Phase 2
        {"ais": 1.0, "te": 0.5, "synergy": 1.0},  # Phase 3
    ]

    for i, lambdas in enumerate(phases):
        print(f"Testing Phase {i+1} Stability (Lambdas: {lambdas})")
        outputs = get_mock_output()
        loss_dict = loss_fn(outputs, targets, lambdas)
        loss = loss_dict["loss"]
        loss.backward()
        
        # Check if any component is NaN
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) and torch.isnan(v):
                print(f"  ⚠ ALERT: {k} is NaN in Phase {i+1}")
        
        print(f"  Loss: {loss.item():.4f} | Grad Norm Check: PASSED")


if __name__ == "__main__":
    diagnose_gradients()
    simulate_phase_transitions()

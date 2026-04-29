import torch
import torch.nn as nn
from typing import cast


class TransferBus(nn.Module):
    """
    TransferBus (Routing Primitive)
    Transfers data from source to target without distortion. Optimized for Transfer Entropy.
    """

    activity_score: torch.Tensor

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim

        # Linear routing matrix
        self.weight = nn.Parameter(torch.randn(target_dim, source_dim) * 0.1)

        # Fidelity gating - can be used to prune the bus during training
        self.register_buffer("activity_score", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Routes the source information to the target.
        """
        # Simple linear propagation (High fidelity)
        return x @ self.weight.T

    def set_activity(self, te_value: float) -> None:
        """Sets the activity score based on the TE validation pass."""
        self.activity_score.copy_(torch.tensor(te_value))

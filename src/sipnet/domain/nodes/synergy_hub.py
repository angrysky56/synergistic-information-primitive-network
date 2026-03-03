import torch
import torch.nn as nn


class SynergyHub(nn.Module):
    """
    SynergyHub (Modification Primitive)
    Integrates feedforward sensory data and contextual memory to extract synergistic logic.
    Optimized for Synergistic Information (PID).
    """
    def __init__(self, ff_dim: int, ctx_dim: int, output_dim: int):
        super().__init__()
        self.ff_dim = ff_dim
        self.ctx_dim = ctx_dim
        self.output_dim = output_dim

        # Dual input compartments
        self.ff_layer = nn.Linear(ff_dim, output_dim)
        self.ctx_layer = nn.Linear(ctx_dim, output_dim)

        # Non-linear integration sector (dendritic synthesis)
        self.synthesis = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, ff_input: torch.Tensor, ctx_input: torch.Tensor) -> torch.Tensor:
        """
        Integrates the two streams via non-linear synthesis.
        """
        ff_features = self.ff_layer(ff_input)
        ctx_features = self.ctx_layer(ctx_input)

        # Concatenate and synthesize
        combined = torch.cat([ff_features, ctx_features], dim=1)
        output = self.synthesis(combined)

        return output

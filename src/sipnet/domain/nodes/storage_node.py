
import torch
import torch.nn as nn


class StorageNode(nn.Module):
    """
    StorageNode (Memory Primitive)
    Uses recurrent dynamics to buffer context passively. Optimized for Active Information Storage.
    """
    def __init__(
        self,
        dim: int,
        input_dim: int,
        spectral_radius: float = 0.99,
        leak_rate: float = 0.1,
        threshold: float = 0.01
    ):
        super().__init__()
        self.dim = dim
        self.leak_rate = leak_rate
        self.threshold = threshold

        # Recurrent weights: Sparse and scaled to spectral radius
        self.recurrent_weights = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self._initialize_recurrent(spectral_radius)

        # Input weights
        self.input_weights = nn.Parameter(torch.randn(dim, input_dim) * 0.5)

        # State (initialized lazily in forward to match batch size)
        self.register_buffer("state", torch.empty(0))

    def _initialize_recurrent(self, spectral_radius: float) -> None:
        """Initializes weights to have a specific spectral radius."""
        with torch.no_grad():
            weight = self.recurrent_weights
            # Compute spectral radius (max absolute eigenvalue)
            # Ensure weight is on CPU for eigvals if needed, though most modern torch versions support it on GPU
            eigenvalues = torch.linalg.eigvals(weight)
            max_eig = torch.max(torch.abs(eigenvalues))
            weight.data *= (spectral_radius / (max_eig + 1e-9))

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """
        Updates the internal state based on input and recurrent connection.
        """
        batch_size = x.shape[0] if x is not None else (self.state.shape[0] if self.state.numel() > 0 else 1)

        # Lazy initialization or resizing of state
        if self.state.numel() == 0 or self.state.shape[0] != batch_size:
            self.state = torch.zeros(batch_size, self.dim, device=self.recurrent_weights.device)

        # IMPORTANT: Detach state to prevent backward through old graphs when training in batches
        prev_state = self.state.detach()


        # Linear integration
        recurrent_component = prev_state @ self.recurrent_weights.T
        input_component = 0.0
        if x is not None:
            input_component = x @ self.input_weights.T

        # Total integrated signal
        total_signal = (1 - self.leak_rate) * prev_state + self.leak_rate * (recurrent_component + input_component)

        # Threshold-linear activation function (ReLU-like)
        new_state = torch.clamp(total_signal - self.threshold, min=0.0)

        self.state = new_state
        return new_state

    def reset_state(self) -> None:
        """Resets memory state."""
        if self.state.numel() > 0:
            self.state.zero_()


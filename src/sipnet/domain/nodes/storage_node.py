"""
StorageNode (Memory Primitive) implementation.
"""

import torch
import torch.nn as nn


class StorageNode(nn.Module):
    """
    StorageNode (Memory Primitive).

    Uses recurrent dynamics to buffer context passively. Optimized for
    Active Information Storage within the synergistic network.
    """

    state: torch.Tensor

    def __init__(
        self,
        dim: int,
        input_dim: int,
        spectral_radius: float = 0.99,
        leak_rate: float = 0.1,
        threshold: float = 0.01,
    ) -> None:
        """
        Initializes the StorageNode.

        Args:
            dim: Dimension of the recurrent state.
            input_dim: Dimension of the input signal.
            spectral_radius: Desired spectral radius for recurrent weight initialization.
            leak_rate: Rate at which the state leaks/updates (0 to 1).
            threshold: Threshold for the threshold-linear activation function.
        """
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
        # We use persistent=False because recurrent state is transient across sequences
        self.register_buffer("state", torch.empty(0), persistent=False)

    def _initialize_recurrent(self, spectral_radius: float) -> None:
        """
        Initializes weights to have a specific spectral radius.

        Args:
            spectral_radius: The target spectral radius.
        """
        with torch.no_grad():
            weight = self.recurrent_weights
            # Compute spectral radius (max absolute eigenvalue)
            # Use torch.linalg.eigvals for modern PyTorch compatibility
            eigenvalues = torch.linalg.eigvals(weight)
            max_eig = torch.max(torch.abs(eigenvalues))
            weight.data *= spectral_radius / (max_eig + 1e-9)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """
        Updates the internal state based on input and recurrent connection.

        Args:
            x: Optional input tensor.

        Returns:
            The updated state tensor.
        """
        batch_size = (
            x.shape[0]
            if x is not None
            else (self.state.shape[0] if self.state.numel() > 0 else 1)
        )

        # Lazy initialization or resizing of state
        if self.state.numel() == 0 or self.state.shape[0] != batch_size:
            new_state = torch.zeros(
                batch_size,
                self.dim,
                device=self.recurrent_weights.device,
                dtype=self.recurrent_weights.dtype,
            )
            self.register_buffer("state", new_state, persistent=False)

        prev_state = self.state

        # Linear integration
        recurrent_component = prev_state @ self.recurrent_weights.T

        # Handle optional input with type-safe initialization
        if x is not None:
            input_component = x @ self.input_weights.T
        else:
            input_component = torch.zeros_like(recurrent_component)

        # Total integrated signal
        total_signal = (1 - self.leak_rate) * prev_state + self.leak_rate * (
            recurrent_component + input_component
        )

        # Threshold-linear activation function (ReLU-like)
        new_state = torch.clamp(total_signal - self.threshold, min=0.0)

        # Update buffer while maintaining its properties
        self.state = new_state
        return new_state

    def reset_state(self) -> None:
        """
        Resets memory state and breaks the backward graph for the next sequence.
        """
        if self.state.numel() > 0:
            # detach() is critical here to break the computational graph between batches
            self.state = self.state.detach().zero_()

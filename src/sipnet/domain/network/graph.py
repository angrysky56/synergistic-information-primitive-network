"""
Synergistic Information Primitive Network (SIP-Net) graph implementation.
"""
from typing import Any

import torch
import torch.nn as nn

from ..common.types import StepOutput
from .sip_layer import SIPLayer


class SIPNet(nn.Module):
    """
    Deep Synergistic Information Primitive Network (Deep SIP-Net).

    A hierarchical graph of stacked SIPLayers that processes information
    through synergistic interactions between storage and transfer components.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_storage_nodes: int = 1,
        num_synergy_hubs: int = 1,
        num_parallel_buses: int = 1,
        use_embedding: bool = False,
    ) -> None:
        """
        Initializes the SIPNet hierarchy.

        Args:
            input_dim: Dimension of the input data or embedding vocabulary size.
            hidden_dim: Dimension of the internal hidden representations.
            output_dim: Dimension of the output logits.
            num_layers: Number of hierarchical SIP layers to stack.
            num_storage_nodes: Number of storage nodes per layer.
            num_synergy_hubs: Number of synergy hubs per layer.
            num_parallel_buses: Number of parallel buses (FF/TD) per layer.
            use_embedding: Whether to use an embedding layer as the sensory encoder.
        """
        super().__init__()
        self.use_embedding = use_embedding

        # 1. Sensory Encoder
        self.encoder: nn.Module
        if self.use_embedding:
            self.encoder = nn.Embedding(
                num_embeddings=input_dim, embedding_dim=hidden_dim
            )
        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)

        # 2. Deep Hierarchical SIP Layers
        self.layers = nn.ModuleList(
            [
                SIPLayer(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_storage_nodes=num_storage_nodes,
                    num_synergy_hubs=num_synergy_hubs,
                    num_parallel_buses=num_parallel_buses,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Output Decoder
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward_step(self, x_t: torch.Tensor) -> StepOutput:
        """
        Processes a single sensory input through the hierarchical SIP layers.

        Args:
            x_t: Input tensor for the current time step.

        Returns:
            A dictionary containing logits and intermediate layer outputs.
        """
        encoded_sensory = self.encoder(x_t)

        current_input = encoded_sensory
        layer_outputs = []

        # Pass data forward through the hierarchy
        for layer in self.layers:
            if not isinstance(layer, SIPLayer):
                continue
            out = layer.forward_step(current_input)
            layer_outputs.append(out)
            # The final representation of Layer L becomes the input for Layer L+1
            current_input = out["final_rep"]

        logits = self.decoder(current_input)

        return {
            "logits": logits,
            "layer_outputs": layer_outputs,  # Export all nested state matrices for loss tracking
            "prev_layer_outputs": None,  # Linked during training
        }

    def forward(self, x_seq: torch.Tensor) -> list[StepOutput]:
        """
        Processes a sequence of sensory inputs over time.

        Args:
            x_seq: Input sequence tensor of shape (batch, sequence_length, ...).

        Returns:
            A list of result dictionaries for each time step.
        """
        seq_len = x_seq.size(1)
        outputs_over_time = []
        self.reset_memory()

        for t in range(seq_len):
            if self.use_embedding:
                x_t = x_seq[:, t]
            else:
                x_t = x_seq[:, t, :]
            outputs_over_time.append(self.forward_step(x_t))

        return outputs_over_time

    def reset_memory(self) -> None:
        """
        Resets the memory state (storage nodes) in all hierarchical layers.
        """
        for layer in self.layers:
            if isinstance(layer, SIPLayer):
                layer.reset_memory()

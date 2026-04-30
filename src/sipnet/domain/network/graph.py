"""
Synergistic Information Primitive Network (SIP-Net) graph implementation.
"""
from typing import Any, Optional

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
        hf_model_name: Optional[str] = None,
        hf_freeze: bool = True,
        dropout: float = 0.1,
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
            hf_model_name: Optional name of a Hugging Face model to use for embeddings.
            hf_freeze: Whether to freeze the Hugging Face model parameters.
            dropout: Dropout rate for noise-resilient projection.
        """
        super().__init__()
        self.use_embedding = use_embedding
        self.hf_model_name = hf_model_name

        # 1. Sensory Encoder
        self.encoder: nn.Module
        if self.hf_model_name:
            from transformers import AutoModel
            model_name = self.hf_model_name
            self.hf_model = AutoModel.from_pretrained(model_name)
            if hf_freeze:
                for param in self.hf_model.parameters():
                    param.requires_grad = False
            # Get embedding dimension from HF model config
            hf_dim = self.hf_model.config.hidden_size
            self.encoder = nn.Sequential(
                nn.Linear(hf_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
        elif self.use_embedding:
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
                 If hf_model_name is set, x_t should be the output of HF model.
        """
        encoded_sensory = self.encoder(x_t)

        current_input = encoded_sensory
        layer_outputs = []

        # Pass data forward through the hierarchy with residual connections
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, SIPLayer):
                continue
            out = layer.forward_step(current_input)
            layer_outputs.append(out)
            
            # Residual connection
            current_input = out["final_rep"] + current_input

        logits = self.decoder(current_input)

        return {
            "logits": logits,
            "layer_outputs": layer_outputs,
            "prev_layer_outputs": None,
        }

    def forward(self, x_seq: torch.Tensor) -> list[StepOutput]:
        """
        Processes a sequence of sensory inputs over time.
        """
        batch_size, seq_len = x_seq.size(0), x_seq.size(1)
        
        # Pre-calculate HF embeddings if necessary
        if self.hf_model_name:
            # We assume x_seq is (batch, seq_len) token IDs
            # This is done once for the whole sequence for efficiency
            with torch.no_grad() if not self.hf_model.training else torch.enable_grad():
                hf_outputs = self.hf_model(x_seq).last_hidden_state # (batch, seq_len, hf_dim)
            x_seq_processed = hf_outputs
        else:
            x_seq_processed = x_seq

        outputs_over_time = []
        self.reset_memory()

        for t in range(seq_len):
            if self.hf_model_name or not self.use_embedding:
                x_t = x_seq_processed[:, t, :]
            else:
                x_t = x_seq_processed[:, t]
            outputs_over_time.append(self.forward_step(x_t))

        return outputs_over_time

    def reset_memory(self) -> None:
        """
        Resets the memory state (storage nodes) in all hierarchical layers.
        """
        for layer in self.layers:
            if isinstance(layer, SIPLayer):
                layer.reset_memory()

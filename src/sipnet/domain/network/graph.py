import torch
import torch.nn as nn

from .sip_layer import SIPLayer


class SIPNet(nn.Module):
    """
    Deep Synergistic Information Primitive Network (Deep SIP-Net)
    A hierarchical graph of stacked SIPLayers.
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
        super().__init__()
        self.use_embedding = use_embedding

        # 1. Sensory Encoder
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

    def forward_step(self, x_t: torch.Tensor) -> dict:
        encoded_sensory = self.encoder(x_t)

        current_input = encoded_sensory
        layer_outputs = []

        # Pass data forward through the hierarchy
        for layer in self.layers:
            out = layer.forward_step(current_input)
            layer_outputs.append(out)
            # The final representation of Layer L becomes the input for Layer L+1
            current_input = out["final_rep"]

        logits = self.decoder(current_input)

        return {
            "logits": logits,
            "layer_outputs": layer_outputs,  # Export all nested state matrices for loss tracking
        }

    def forward(self, x_seq: torch.Tensor) -> list[dict]:
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
        for layer in self.layers:
            layer.reset_memory()

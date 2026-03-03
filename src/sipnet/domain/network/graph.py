
import torch
import torch.nn as nn

from ..nodes.storage_node import StorageNode
from ..nodes.synergy_hub import SynergyHub
from ..nodes.transfer_bus import TransferBus


class SIPNet(nn.Module):
    """
    Synergistic Information Primitive Network (SIP-Net)
    A heterogeneous graph of specialized computational modules.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_storage_nodes: int = 1,
        num_synergy_hubs: int = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 1. Input Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # 2. Information Processing Primitives
        # Storage Nodes (Memory)
        self.storage_nodes = nn.ModuleList([
            StorageNode(hidden_dim, hidden_dim) for _ in range(num_storage_nodes)
        ])

        # Transfer Buses (Routing)
        # Feedforward Bus: Encoder -> Synergy Hub
        self.ff_bus = TransferBus(hidden_dim, hidden_dim)
        # Top-Down Bus: Storage Node -> Synergy Hub
        self.td_bus = TransferBus(hidden_dim, hidden_dim)

        # Synergy Hubs (Integration/Logic)
        self.synergy_hubs = nn.ModuleList([
            SynergyHub(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_synergy_hubs)
        ])

        # 3. Output Decoder
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward_step(self, x_t: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Operational pipeline: Encoding -> Buffering -> Routing Query -> Integration.
        """
        # Step 1: Sensory Encoding
        encoded = self.encoder(x_t)

        # Step 2: Buffering Core Context
        # (In a real sequence, this would be conditional. Here we buffer the current state)
        storage_outputs = []
        for node in self.storage_nodes:
            storage_outputs.append(node(encoded))

        # Aggregate storage context (averaging for simplicity in bivariate PID)
        context_state = torch.mean(torch.stack(storage_outputs), dim=0)

        # Step 3: Routing via Transfer Buses
        ff_signal = self.ff_bus(encoded)
        ctx_signal = self.td_bus(context_state)

        # Step 4: Synergistic Computation
        hub_outputs = []
        for hub in self.synergy_hubs:
            hub_outputs.append(hub(ff_signal, ctx_signal))

        final_rep = torch.mean(torch.stack(hub_outputs), dim=0)

        # Step 5: Prediction
        logits = self.decoder(final_rep)

        # Return internal states for information-theoretic regularization
        return {
            "logits": logits,
            "encoded": encoded,
            "ff_signal": ff_signal,
            "ctx_signal": ctx_signal,
            "context_state": context_state,
            "final_rep": final_rep
        }

    def forward(self, x_seq: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """
        Unrolls the computation over the temporal sequence dimension.
        x_seq expected shape: [batch_size, seq_len, input_dim]
        """
        seq_len = x_seq.size(1)
        outputs_over_time = []

        # Ensure fresh memory at the start of every sequence
        self.reset_memory()

        for t in range(seq_len):
            # Extract input for the current timestep: shape [batch_size, input_dim]
            x_t = x_seq[:, t, :]

            # Step the graph
            step_output = self.forward_step(x_t)
            outputs_over_time.append(step_output)

        return outputs_over_time

    def reset_memory(self) -> None:
        """Resets all storage nodes."""
        for node in self.storage_nodes:
            node.reset_state()

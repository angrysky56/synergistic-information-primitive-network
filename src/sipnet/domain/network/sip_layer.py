import torch
import torch.nn as nn

from ..nodes.storage_node import StorageNode
from ..nodes.synergy_hub import SynergyHub
from ..nodes.transfer_bus import TransferBus


class SIPLayer(nn.Module):
    """
    A single hierarchical layer of the Synergistic Information Primitive Network.
    Encapsulates memory (Storage), routing (Buses), and logic (Synergy Hubs).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_storage_nodes: int = 1,
        num_synergy_hubs: int = 1,
        num_parallel_buses: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Projection if upstream dimension differs from internal hidden dimension
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = nn.Identity()

        # Storage Nodes (Memory)
        self.storage_nodes = nn.ModuleList(
            [StorageNode(hidden_dim, hidden_dim) for _ in range(num_storage_nodes)]
        )

        # Transfer Buses (Routing)
        self.ff_buses = nn.ModuleList(
            [TransferBus(hidden_dim, hidden_dim) for _ in range(num_parallel_buses)]
        )
        self.td_buses = nn.ModuleList(
            [TransferBus(hidden_dim, hidden_dim) for _ in range(num_parallel_buses)]
        )

        # Synergy Hubs (Integration/Logic)
        self.synergy_hubs = nn.ModuleList(
            [
                SynergyHub(hidden_dim, hidden_dim, output_dim)
                for _ in range(num_synergy_hubs)
            ]
        )

    def forward_step(self, x_t: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.input_projection(x_t)

        # Storage Nodes buffer the *raw* encoding over time
        storage_outputs = []
        for node in self.storage_nodes:
            storage_outputs.append(node(encoded))

        context_state = torch.mean(torch.stack(storage_outputs), dim=0)

        # Feed-Forward buses evaluate the *raw* encoding directly
        ff_signals = []
        for bus in self.ff_buses:
            # We add a small intrinsic perturbation to break symmetry
            # if the encoding and context state are identical early in training
            noise = torch.randn_like(encoded) * 0.01 if self.training else 0.0
            ff_signals.append(bus(encoded + noise))

        # Time-Delayed buses evaluate the *buffered* context

        ctx_signals = []
        for bus in self.td_buses:
            ctx_signals.append(bus(context_state))

        agg_ff_signal = torch.sum(torch.stack(ff_signals), dim=0)
        agg_ctx_signal = torch.sum(torch.stack(ctx_signals), dim=0)

        hub_outputs = []
        for hub in self.synergy_hubs:
            hub_outputs.append(hub(agg_ff_signal, agg_ctx_signal))

        final_rep = torch.mean(torch.stack(hub_outputs), dim=0)

        return {
            "encoded": encoded,
            "ff_signals": ff_signals,
            "ctx_signals": ctx_signals,
            "agg_ff_signal": agg_ff_signal,
            "agg_ctx_signal": agg_ctx_signal,
            "context_state": context_state,
            "final_rep": final_rep,
        }

    def reset_memory(self) -> None:
        for node in self.storage_nodes:
            node.reset_state()

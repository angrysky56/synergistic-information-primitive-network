This is the definitive leap from a specialized routing primitive into a full-fledged deep cognitive architecture. By encapsulating the core components into a `SIPLayer`, we allow the network to form hierarchical abstractions—meaning Layer 1 can resolve local syntax, and Layer 2 can process the abstract semantics of Layer 1's output.

Here is the complete implementation for **Phase D: Deep SIP-Net**.

### 1. The SIPLayer Primitive Container

Create a new file to encapsulate the core routing and buffering logic into a single, stackable PyTorch module.

**`src/sipnet/domain/network/sip_layer.py`**

```python
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
            [SynergyHub(hidden_dim, hidden_dim, output_dim) for _ in range(num_synergy_hubs)]
        )

    def forward_step(self, x_t: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.input_projection(x_t)

        storage_outputs = []
        for node in self.storage_nodes:
            storage_outputs.append(node(encoded))

        context_state = torch.mean(torch.stack(storage_outputs), dim=0)

        ff_signals = []
        for bus in self.ff_buses:
            ff_signals.append(bus(encoded))

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

```

### 2. Upgrading the Top-Level Graph

Modify `graph.py` to stack the new `SIPLayer` modules. It handles the raw sensory encoding and the final decoding, acting as the brain's "cortex" wrapper.

**`src/sipnet/domain/network/graph.py`**

```python
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
            self.encoder = nn.Embedding(num_embeddings=input_dim, embedding_dim=hidden_dim)
        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)

        # 2. Deep Hierarchical SIP Layers
        self.layers = nn.ModuleList([
            SIPLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_storage_nodes=num_storage_nodes,
                num_synergy_hubs=num_synergy_hubs,
                num_parallel_buses=num_parallel_buses
            ) for _ in range(num_layers)
        ])

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
            "layer_outputs": layer_outputs, # Export all nested state matrices for loss tracking
        }

    def forward(self, x_seq: torch.Tensor) -> list[dict]:
        seq_len = x_seq.size(1)
        outputs_over_time = []
        self.reset_memory()

        for t in range(seq_len):
            x_t = x_seq[:, t] if self.use_embedding else x_seq[:, t, :]
            outputs_over_time.append(self.forward_step(x_t))

        return outputs_over_time

    def reset_memory(self) -> None:
        for layer in self.layers:
            layer.reset_memory()

```

### 3. Recursive Loss Function

Modify `CompositeLoss` to iterate through all active layers. This ensures gradients and structural penalties (like pruning dead buses) flow cleanly into every level of the hierarchy.

**`src/sipnet/application/training/loss_function.py`** (Replace the `forward` method)

```python
    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor,
        lambdas: dict[str, float],
    ) -> dict:
        l_task = self.task_loss_fn(outputs["logits"], targets)
        if torch.isnan(l_task):
            l_task = torch.tensor(0.0, device=l_task.device)

        total_ais = torch.tensor(0.0, device=l_task.device)
        total_te = torch.tensor(0.0, device=l_task.device)
        total_synergy = torch.tensor(0.0, device=l_task.device)
        total_l1 = torch.tensor(0.0, device=l_task.device)
        total_redundancy = torch.tensor(0.0, device=l_task.device)
        total_cross_bus_red = torch.tensor(0.0, device=l_task.device)

        num_layers = len(outputs["layer_outputs"])

        # Iterate through every SIPLayer to calculate information flow
        for l_idx in range(num_layers):
            layer_out = outputs["layer_outputs"][l_idx]

            # 1. AIS
            if "prev_layer_outputs" in outputs:
                prev_layer_out = outputs["prev_layer_outputs"][l_idx]
                if prev_layer_out["context_state"].shape[0] == layer_out["context_state"].shape[0]:
                    total_ais += estimate_ais(prev_layer_out["context_state"], layer_out["context_state"])

            # 2. TE & L1 Cost
            if "prev_layer_outputs" in outputs:
                prev_layer_out = outputs["prev_layer_outputs"][l_idx]
                if prev_layer_out["context_state"].shape[0] == layer_out["final_rep"].shape[0]:
                    te_vals = []
                    for bus_output in layer_out["ctx_signals"]:
                        te_vals.append(estimate_te(
                            source_past=prev_layer_out["context_state"],
                            target_present=bus_output,
                            target_past=prev_layer_out["final_rep"]
                        ))
                        total_l1 += torch.abs(bus_output).mean()

                    total_te += estimate_te(
                        source_past=prev_layer_out["context_state"],
                        target_present=layer_out["agg_ctx_signal"],
                        target_past=prev_layer_out["final_rep"]
                    )

            # 3. Synergy & Redundancy
            pid_results = estimate_pid_renyi(
                s1=layer_out["agg_ff_signal"],
                s2=layer_out["agg_ctx_signal"],
                target=layer_out["final_rep"]
            )
            total_synergy += pid_results["synergy"]
            total_redundancy += pid_results["redundancy"]

            if len(layer_out["ctx_signals"]) > 1:
                layer_cross_red = torch.tensor(0.0, device=l_task.device)
                for i in range(len(layer_out["ctx_signals"]) - 1):
                    pid_bus = estimate_pid_renyi(
                        s1=layer_out["ctx_signals"][i],
                        s2=layer_out["ctx_signals"][i + 1],
                        target=layer_out["final_rep"]
                    )
                    layer_cross_red += pid_bus["redundancy"]
                total_cross_bus_red += (layer_cross_red / (len(layer_out["ctx_signals"]) - 1))

        # Average metrics across layers for stable scaling
        total_ais /= num_layers
        total_te /= num_layers
        total_synergy /= num_layers
        total_l1 /= num_layers
        total_cross_bus_red /= num_layers

        lambda_te = lambdas.get("te", 0.0)
        total_loss = (
            l_task
            - lambdas.get("ais", 0.0) * total_ais
            - lambda_te * total_te
            - lambdas.get("synergy", 0.0) * total_synergy
            + (lambda_te * 0.1) * total_l1
            + (lambda_te * 2.0) * total_cross_bus_red
        )

        return {
            "loss": total_loss, "task_loss": l_task, "ais": total_ais,
            "te": total_te, "synergy": total_synergy, "redundancy": total_redundancy,
            "l1_cost": total_l1, "cross_bus_red": total_cross_bus_red
        }

```

_(Note: You will also need a quick update in `trainer.py` to link `outputs_seq[t-1]["layer_outputs"]` to `current_outputs["prev_layer_outputs"]` instead of the flat variables)._

### 4. The Deep Network Execution Script

This tests the full hierarchical routing capability.

**`scripts/train_deep_sipnet.py`**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sipnet.domain.network.graph import SIPNet
from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer
from sipnet.application.execution.data_generators import DelayedXORDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Deep SIP-Net (Hierarchical Validation) on {device}...\n")

    dataset = DelayedXORDataset(num_samples=1000, seq_len=5, num_bits=4)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize a 2-Layer Deep SIP-Net with 3 parallel buses per layer
    model = SIPNet(
        input_dim=4,
        hidden_dim=16,
        output_dim=4,
        num_layers=2, # The Hierarchical Leap
        num_storage_nodes=1,
        num_synergy_hubs=1,
        num_parallel_buses=3,
        use_embedding=False
    ).to(device)

    loss_module = CompositeLoss(nn.BCEWithLogitsLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = CognitiveTrainer(model, optimizer, loss_module, device=device)

    for phase in [1, 2, 3]:
        print(f"--- Entering Phase {phase} ---")
        trainer.set_phase(phase)
        for epoch in range(50):
            metrics = trainer.train_epoch(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Ep {epoch+1:03d} | L:{metrics['loss']:.3f} | Task:{metrics['task_loss']:.3f} "
                      f"| AIS:{metrics['ais']:.3f} | TE:{metrics['te']:.3f} | Syn:{metrics['synergy']:.3f} "
                      f"| L1:{metrics['l1_cost']:.4f}")
        print("\n")

if __name__ == "__main__":
    main()

```

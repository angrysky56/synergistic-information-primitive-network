import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sipnet.domain.network.graph import SIPNet
from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer
from sipnet.application.execution.nlp_generators import SequentialNLPDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing SIP-Net Multi-Bus Routing Validation on {device}...\n")

    # Dataset
    num_samples = 2000
    dataset = SequentialNLPDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Network
    vocab_size = dataset.vocab_size
    hidden_dim = 16
    num_buses = 5

    model = SIPNet(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        output_dim=vocab_size, # Predict word class
        num_storage_nodes=1,
        num_synergy_hubs=1,
        num_parallel_buses=num_buses,
        use_embedding=True
    ).to(device)

    # CrossEntropyLoss automatically ignores target index 0 which corresponds to our <PAD>
    task_loss = nn.CrossEntropyLoss(ignore_index=0)
    loss_module = CompositeLoss(task_loss_fn=task_loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = CognitiveTrainer(model, optimizer, loss_module, device=device)

    epochs_per_phase = 50

    for phase in [1, 2, 3]:
        print(f"\n--- Entering Phase {phase} ---")
        trainer.set_phase(phase)

        for epoch in range(1, epochs_per_phase + 1):
            metrics = trainer.train_epoch(dataloader)

            if epoch % 10 == 0:
                # Format bus outputs distinctively to watch dynamic pruning logic natively!
                bus_te_strs = []
                for b in range(num_buses):
                    bus_key = f"te_buses_{b}"
                    if bus_key in metrics:
                        bus_te_strs.append(f"B{b}:{metrics[bus_key]:.3f}")

                bus_str = " | ".join(bus_te_strs) if bus_te_strs else f"TE:{metrics['te']:.3f}"

                print(f"Ep {epoch:03d} | L:{metrics['loss']:.3f} | Task:{metrics['task_loss']:.3f} | "
                      f"AIS:{metrics['ais']:.3f} | {bus_str} | Syn:{metrics['synergy']:.3f} | "
                      f"L1:{metrics.get('l1_cost', 0.0):.4f} | Red:{metrics.get('redundancy', 0.0):.3f}")

if __name__ == "__main__":
    main()

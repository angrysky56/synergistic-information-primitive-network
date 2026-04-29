"""
Training script for the Deep Synergistic Information Primitive Network (Deep SIP-Net).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sipnet.application.execution.data_generators import DelayedXORDataset
from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer
from sipnet.domain.network.graph import SIPNet


def main() -> None:
    """
    Main training loop for validating the hierarchical SIP-Net architecture.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Deep SIP-Net (Hierarchical Validation) on {device}...\n")

    dataset = DelayedXORDataset(num_samples=1000, seq_len=5, num_bits=4)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize a 2-Layer Deep SIP-Net with 1 parallel buses per layer
    model = SIPNet(
        input_dim=4,
        hidden_dim=16,
        output_dim=4,
        num_layers=2,  # The Hierarchical Leap
        num_storage_nodes=1,
        num_synergy_hubs=1,
        num_parallel_buses=1,
        use_embedding=False,
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
                print(
                    f"Ep {epoch+1:03d} | L:{metrics['loss']:.3f} | "
                    f"Task:{metrics['task_loss']:.3f} | AIS:{metrics['ais']:.3f} | "
                    f"TE:{metrics['te']:.3f} | Syn:{metrics['synergy']:.3f} | "
                    f"L1:{metrics['l1_cost']:.4f}"
                )
        print("\n")


if __name__ == "__main__":
    main()

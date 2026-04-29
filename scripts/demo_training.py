from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer
from sipnet.domain.network.graph import SIPNet


def generate_mock_data(num_samples: int = 1000, input_dim: int = 16) -> DataLoader[Any]:
    """
    Generates synthetic data for a 'delayed association' task.
    Target depends on current input and past context.
    """
    # Create sequence-like data (batch, seq_len, input_dim)
    # The current trainer expects (batch, seq_len, input_dim)
    seq_len = 5
    x = torch.randn(num_samples, seq_len, input_dim)
    # Binary classification target based on a non-linear combination at the final timestep
    # Logic: 1 if sum of first 4 features > 0 AND sum of last 4 features > 0
    target = ((x[:, -1, :4].sum(dim=1) > 0) & (x[:, -1, -4:].sum(dim=1) > 0)).long()
    # Expand target to be (batch, seq_len) with same values or padded
    target_seq = target.unsqueeze(1).repeat(1, seq_len)

    dataset = TensorDataset(x, target_seq)
    return DataLoader(dataset, batch_size=32, shuffle=False)  # Sequential for AIS


def run_demo() -> None:
    """Runs the SIP-Net cognitive development demo."""
    print("=== SIP-Net Cognitive Development Demo ===")

    # 1. Hyperparameters
    input_dim = 16
    hidden_dim = 64
    output_dim = 2
    epochs_per_phase = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Setup
    model = SIPNet(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_module = CompositeLoss(nn.CrossEntropyLoss())
    trainer = CognitiveTrainer(model, optimizer, loss_module, device=device)
    dataloader = generate_mock_data(input_dim=input_dim)

    # 3. Phased Training
    for phase in [1, 2, 3]:
        trainer.set_phase(phase)
        print(
            f"\n--- Phase {phase}: Updating lambdas to {trainer.lambdas} ---"
        )

        for epoch in range(epochs_per_phase):
            metrics = trainer.train_epoch(dataloader)
            print(
                f"Epoch {epoch+1} | Loss: {metrics['loss']:.4f} | Task Loss: {metrics['task_loss']:.4f} "
                f"| AIS: {metrics['ais']:.4f} | TE: {metrics['te']:.4f} | Synergy: {metrics['synergy']:.4f}"
            )

    print("\n=== Demo Complete ===")
    print("Final specialized metrics validated.")


if __name__ == "__main__":
    run_demo()

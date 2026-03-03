import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sipnet.domain.network.graph import SIPNet
from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer

def generate_mock_data(num_samples: int = 1000, input_dim: int = 16):
    """
    Generates synthetic data for a 'delayed association' task.
    Target depends on current input and past context.
    """
    x = torch.randn(num_samples, input_dim)
    # Binary classification target based on a non-linear combination
    # Logic: 1 if sum of first 4 features > 0 AND sum of last 4 features > 0
    target = ((x[:, :4].sum(dim=1) > 0) & (x[:, -4:].sum(dim=1) > 0)).long()

    dataset = TensorDataset(x, target)
    return DataLoader(dataset, batch_size=32, shuffle=False) # Sequential for AIS

def run_demo():
    print("=== SIP-Net Cognitive Development Demo ===")

    # 1. Hyperparameters
    input_dim = 16
    hidden_dim = 64
    output_dim = 2
    epochs_per_phase = 3

    # 2. Setup
    model = SIPNet(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_module = CompositeLoss(nn.CrossEntropyLoss())
    trainer = CognitiveTrainer(model, optimizer, loss_module)
    dataloader = generate_mock_data()

    # 3. Phased Training
    for phase in [1, 2, 3]:
        print(f"\n--- Phase {phase}: Updating lambdas to {trainer.lambdas if phase > 1 else 'Initialization'} ---")
        trainer.set_phase(phase)

        for epoch in range(epochs_per_phase):
            metrics = trainer.train_epoch(dataloader)
            print(f"Epoch {epoch+1} | Loss: {metrics['loss']:.4f} | Task Loss: {metrics['task_loss']:.4f} | AIS: {metrics['ais']:.4f} | TE: {metrics['te']:.4f} | Synergy: {metrics['synergy']:.4f}")

    print("\n=== Demo Complete ===")
    print("Final specialized metrics validated.")

if __name__ == "__main__":
    run_demo()

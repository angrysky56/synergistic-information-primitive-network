import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sipnet.domain.network.graph import SIPNet
from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer
from sipnet.application.execution.data_generators import DelayedXORDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing SIP-Net Temporal Validation on {device}...\n")

    # Hyperparameters
    num_bits = 4
    seq_len = 5
    hidden_dim = 16
    batch_size = 32
    epochs_per_phase = 50

    # Dataset & Loader
    dataset = DelayedXORDataset(num_samples=1000, seq_len=seq_len, num_bits=num_bits)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model Initialization
    model = SIPNet(
        input_dim=num_bits,
        hidden_dim=hidden_dim,
        output_dim=num_bits,
        num_storage_nodes=2,
        num_synergy_hubs=1
    ).to(device)

    # Optimization
    # Use BCEWithLogitsLoss since XOR is a multi-label binary classification task
    task_loss_fn = nn.BCEWithLogitsLoss()
    loss_module = CompositeLoss(task_loss_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = CognitiveTrainer(model, optimizer, loss_module, device=device)

    # Execute Cognitive Phases
    for phase in [1, 2, 3]:
        print(f"--- Entering Phase {phase} ---")
        trainer.set_phase(phase)

        for epoch in range(epochs_per_phase):
            metrics = trainer.train_epoch(dataloader)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:03d} | Loss: {metrics['loss']:.4f} | Task: {metrics['task_loss']:.4f} "
                      f"| AIS: {metrics['ais']:.4f} | TE: {metrics['te']:.4f} | Syn: {metrics['synergy']:.4f}")
        print("\n")

if __name__ == "__main__":
    main()

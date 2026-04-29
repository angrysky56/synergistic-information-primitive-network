import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sipnet.application.execution.nlp_generators import (
    SequentialNLPDataset,
    nlp_collate_fn,
)
from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer
from sipnet.domain.network.graph import SIPNet


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing SIP-Net NLP Coreference Validation on {device}...\n")

    # Dataset
    num_samples = 2000
    dataset = SequentialNLPDataset(num_samples=num_samples)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=nlp_collate_fn
    )

    # Network
    vocab_size = dataset.vocab_size
    hidden_dim = 16

    model = SIPNet(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        output_dim=vocab_size,
        num_storage_nodes=1,
        num_synergy_hubs=1,
        use_embedding=True,
    ).to(device)

    # CrossEntropyLoss automatically ignores target index 0 which corresponds to our <PAD>
    task_loss = nn.CrossEntropyLoss(ignore_index=0)
    loss_module = CompositeLoss(task_loss_fn=task_loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = CognitiveTrainer(model, optimizer, loss_module, device=device)

    epochs_per_phase = 20

    for phase in [1, 2, 3]:
        print(f"\n--- Entering Phase {phase} ---")
        trainer.set_phase(phase)

        for epoch in range(1, epochs_per_phase + 1):
            metrics = trainer.train_epoch(dataloader)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:03d} | Loss: {metrics['loss']:.4f} | Task: {metrics['task_loss']:.4f} | "
                    f"Acc: {metrics.get('accuracy', 0.0):.4f} | "
                    f"AIS: {metrics['ais']:.4f} | TE: {metrics['te']:.4f} | Syn: {metrics['synergy']:.4f}"
                )

    # Save finalized model
    torch.save(model.state_dict(), "best_model.pt")
    print("\nModel saved to best_model.pt")


if __name__ == "__main__":
    main()

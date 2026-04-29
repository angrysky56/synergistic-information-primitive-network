import time

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


def diag() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SequentialNLPDataset(num_samples=64)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=nlp_collate_fn
    )

    model = SIPNet(
        input_dim=dataset.vocab_size,
        hidden_dim=16,
        output_dim=dataset.vocab_size,
        num_storage_nodes=1,
        num_synergy_hubs=1,
        use_embedding=True,
    ).to(device)

    task_loss = nn.CrossEntropyLoss(ignore_index=0)
    loss_module = CompositeLoss(task_loss_fn=task_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = CognitiveTrainer(model, optimizer, loss_module, device=device)
    trainer.set_phase(1)

    print("Starting single batch diagnostic...")
    start = time.time()
    metrics = trainer.train_epoch(dataloader)
    end = time.time()
    print(f"Batch completed in {end - start:.2f} seconds.")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    diag()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any

from sipnet.infrastructure.data.ontonotes_loader import CoNLL2012Dataset, ontonotes_collate_fn
from sipnet.application.execution.coref_head import CorefMentionRankingHead
from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer
from sipnet.domain.network.graph import SIPNet

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing SIP-Net OntoNotes Coreference Training on {device}...")

    # For sanity check, we'll use a very small mock dataset path or handle missing file
    data_path = "data/ontonotes/train.conll"
    dataset = CoNLL2012Dataset(data_path, tokenizer_name="bert-base-uncased")
    
    if len(dataset) == 0:
        print("No data found at data/ontonotes/train.conll. Please ensure OntoNotes is available.")
        # Create a dummy entry for dry-run if needed
        return

    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=ontonotes_collate_fn
    )

    hidden_dim = 64
    model = SIPNet(
        input_dim=0, # Not used when hf_model_name is set
        hidden_dim=hidden_dim,
        output_dim=hidden_dim, # We'll use the final rep for the head
        num_layers=3,
        hf_model_name="bert-base-uncased",
        hf_freeze=True
    ).to(device)

    coref_head = CorefMentionRankingHead(hidden_dim=hidden_dim).to(device)
    
    # Composite Loss with Inter-Layer Synergy
    task_loss = nn.BCEWithLogitsLoss() # Simplified for mention ranking
    loss_module = CompositeLoss(task_loss_fn=task_loss)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(coref_head.parameters()), lr=1e-4)
    trainer = CognitiveTrainer(model, optimizer, loss_module, device=device)

    # Training loop
    for phase in [1, 2, 3]:
        print(f"\n--- Phase {phase} ---")
        trainer.set_phase(phase)
        
        for epoch in range(1, 6): # Short epochs for verification
            metrics = trainer.train_epoch(dataloader)
            print(f"Epoch {epoch} | Loss: {metrics['loss']:.4f} | Inter-Layer Syn: {metrics.get('inter_layer_syn', 0):.4f}")

if __name__ == "__main__":
    main()

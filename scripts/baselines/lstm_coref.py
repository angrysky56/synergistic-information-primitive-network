import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from sipnet.application.execution.nlp_generators import SequentialNLPDataset, nlp_collate_fn

class LSTMCorefBaseline(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        logits = self.fc(output)
        return logits

def train_baseline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training LSTM Coref Baseline on {device}...")

    # Dataset
    num_samples = 2000
    dataset = SequentialNLPDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=nlp_collate_fn)

    # Model
    model = LSTMCorefBaseline(dataset.vocab_size, 16, dataset.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Training loop
    for epoch in range(1, 101):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        model.train()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            # Flatten for CrossEntropy: (B*T, C)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy at pronoun timestep (where y != 0)
            with torch.no_grad():
                mask = y != 0
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    correct += (preds[mask] == y[mask]).sum().item()
                    total += mask.sum().item()

        if epoch % 10 == 0:
            acc = correct / total if total > 0 else 0
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss/len(dataloader):.4f} | Coref Acc: {acc:.4f}")

if __name__ == "__main__":
    train_baseline()

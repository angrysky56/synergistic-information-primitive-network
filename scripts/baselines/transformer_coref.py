import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import math

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from sipnet.application.execution.nlp_generators import SequentialNLPDataset, nlp_collate_fn

class TransformerCorefBaseline(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, output_dim: int, nhead: int = 2, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create mask for padding
        padding_mask = (x == 0)
        
        embedded = self.embedding(x) * math.sqrt(self.hidden_dim)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        logits = self.fc(output)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return x

def train_baseline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Transformer Coref Baseline on {device}...")

    # Dataset
    num_samples = 2000
    dataset = SequentialNLPDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=nlp_collate_fn)

    # Model
    model = TransformerCorefBaseline(dataset.vocab_size, 16, dataset.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
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

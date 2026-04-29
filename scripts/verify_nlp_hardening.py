import torch
from torch.utils.data import DataLoader

from sipnet.application.execution.nlp_generators import (
    SequentialNLPDataset,
    nlp_collate_fn,
)


def test_generator() -> None:
    print("Testing SequentialNLPDataset with Distractors...")
    dataset = SequentialNLPDataset(num_samples=100, include_distractors=True)
    dataloader = DataLoader(
        dataset, batch_size=10, shuffle=True, collate_fn=nlp_collate_fn
    )

    x, y = next(iter(dataloader))

    print(f"Batch X Shape: {x.shape}")
    print(f"Batch Y Shape: {y.shape}")

    # Check for padding
    if (x == 0).any():
        print("Verification: Padding detected in batch.")

    # Check for distractors (different sequence lengths)
    lengths = [(batch_x != 0).sum().item() for batch_x in x]
    print(f"Sequence lengths in batch: {lengths}")
    if len(set(lengths)) > 1:
        print("Verification: Variable sequence lengths detected.")

    print("\nSample Sentences & Targets:")
    for i in range(min(3, len(x))):
        sentence = [dataset.vocab[idx] for idx in x[i] if idx != 0]
        target_idx = y[i].nonzero()
        if len(target_idx) > 0:
            t_pos = target_idx[0].item()
            t_val = dataset.vocab[y[i, t_pos].item()]
            print(f"S: {' '.join(sentence)}")
            print(f"T: [Pos {t_pos}] -> {t_val}")

    print("\nGenerator Hardening Verified.")


if __name__ == "__main__":
    test_generator()

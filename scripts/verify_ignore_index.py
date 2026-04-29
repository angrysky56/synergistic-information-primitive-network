import torch
import torch.nn as nn


def test_ignore_index() -> None:
    print("Verifying ignore_index=0 behavior...")
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Batch size 1, Seq len 3, Classes 5
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0, 0.0, 0.0],  # Max at index 0 (PAD)
                [0.0, 10.0, 0.0, 0.0, 0.0],  # Max at index 1
                [0.0, 0.0, 10.0, 0.0, 0.0],
            ]
        ]
    )  # Max at index 2

    # Targets: 0 (PAD), 1, 0 (PAD)
    targets = torch.tensor([[0, 1, 0]])

    # Loss should only be calculated for index 1
    loss = criterion(logits.view(-1, 5), targets.view(-1))

    # Logits[1, 1] is 10.0, others are 0.0.
    # Softmax at index 1: exp(10) / (exp(10) + 4*exp(0)) approx 1.
    # -log(1) approx 0.

    print(f"Calculated Loss: {loss.item():.6f}")

    if loss.item() < 0.1:
        print("Verification: ignore_index=0 correctly ignored PAD positions.")
    else:
        print("Verification FAIL: Loss seems too high.")


if __name__ == "__main__":
    test_ignore_index()

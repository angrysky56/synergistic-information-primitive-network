import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sipnet.domain.network.graph import SIPNet
from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer
from sipnet.application.execution.data_generators import DelayedXORDataset


def test_sipnet_end_to_end():
    # Setup model
    input_dim = 4
    hidden_dim = 16
    output_dim = 4
    model = SIPNet(input_dim, hidden_dim, output_dim)

    # Setup data (using new temporal sequence generator)
    dataset = DelayedXORDataset(num_samples=8, seq_len=3, num_bits=input_dim)
    dataloader = DataLoader(dataset, batch_size=4)

    # Setup training modules
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    task_loss_fn = nn.BCEWithLogitsLoss()
    loss_module = CompositeLoss(task_loss_fn)
    trainer = CognitiveTrainer(model, optimizer, loss_module)

    # 1. Verify forward pass (Sequence unrolling)
    # Grab one batch sequence
    x_seq, _ = next(iter(dataloader))
    outputs_seq = model(x_seq)

    # Check that it unrolled correctly over seq_len (3)
    assert len(outputs_seq) == 3

    # Check the contents of a single timestep output
    step_out = outputs_seq[0]
    assert "logits" in step_out
    assert "encoded" in step_out
    assert "ff_signal" in step_out
    assert "ctx_signal" in step_out
    assert step_out["logits"].shape == (4, output_dim)

    # 2. Verify training step (Phase 1 BPTT)
    trainer.set_phase(1)
    metrics = trainer.train_epoch(dataloader)
    assert "loss" in metrics
    assert "task_loss" in metrics
    assert "synergy" in metrics
    assert metrics["loss"] is not None

    # 3. Verify phase transition (Phase 3 BPTT)
    trainer.set_phase(3)
    assert trainer.lambdas["synergy"] == 1.0
    metrics_p3 = trainer.train_epoch(dataloader)
    assert metrics_p3["loss"] is not None


def test_sipnet_differentiability():
    model = SIPNet(10, 20, 2)

    # X needs to be a sequence now [batch_size, seq_len, input_dim]
    x_seq = torch.randn(5, 3, 10, requires_grad=True)
    # Target needs to be a sequence [batch_size, seq_len, target_dim]
    y_seq = torch.randn(5, 3, 2) # BCE needs float targets

    task_loss_fn = nn.BCEWithLogitsLoss()
    loss_module = CompositeLoss(task_loss_fn)
    lambdas = {"ais": 1.0, "te": 1.0, "synergy": 1.0}

    # BPTT unroll
    outputs_seq = model(x_seq)

    total_loss = 0.0
    for t in range(len(outputs_seq)):
        current_outputs = outputs_seq[t]

        if t > 0:
            current_outputs["prev_encoded"] = outputs_seq[t-1]["encoded"]
            current_outputs["prev_context_state"] = outputs_seq[t-1]["context_state"]

        current_targets = y_seq[:, t, :]
        loss_dict = loss_module(current_outputs, current_targets, lambdas)

        if t < len(outputs_seq) - 1:
            loss_dict["loss"] = loss_dict["loss"] - loss_dict["task_loss"]

        total_loss += loss_dict["loss"]

    # Backward pass over unrolled sequence
    total_loss.backward()

    # Check if gradients flowed to the encoder
    assert model.encoder.weight.grad is not None
    # Check if gradients flowed to Storage Nodes
    assert model.storage_nodes[0].recurrent_weights.grad is not None
    # Check if gradients flowed to Synergy Hub
    assert model.synergy_hubs[0].ff_layer.weight.grad is not None

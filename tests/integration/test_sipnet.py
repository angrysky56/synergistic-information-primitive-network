import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sipnet.domain.network.graph import SIPNet
from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer

def test_sipnet_end_to_end():
    # Setup model
    input_dim = 16
    hidden_dim = 32
    output_dim = 2
    model = SIPNet(input_dim, hidden_dim, output_dim)

    # Setup data
    x = torch.randn(8, input_dim)
    y = torch.randint(0, output_dim, (8,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=4)

    # Setup training modules
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    task_loss_fn = nn.CrossEntropyLoss()
    loss_module = CompositeLoss(task_loss_fn)
    trainer = CognitiveTrainer(model, optimizer, loss_module)

    # 1. Verify forward pass
    outputs = model(x)
    assert "logits" in outputs
    assert "encoded" in outputs
    assert "ff_signal" in outputs
    assert "ctx_signal" in outputs
    assert outputs["logits"].shape == (8, output_dim)

    # 2. Verify training step (Phase 1)
    trainer.set_phase(1)
    metrics = trainer.train_epoch(dataloader)
    assert "loss" in metrics
    assert "task_loss" in metrics
    assert "synergy" in metrics
    assert metrics["loss"] is not None

    # 3. Verify phase transition (Phase 3)
    trainer.set_phase(3)
    assert trainer.lambdas["synergy"] == 1.0
    metrics_p3 = trainer.train_epoch(dataloader)
    assert metrics_p3["loss"] is not None

def test_sipnet_differentiability():
    model = SIPNet(10, 20, 2)
    x = torch.randn(5, 10, requires_grad=True)
    y = torch.randint(0, 2, (5,))

    task_loss_fn = nn.CrossEntropyLoss()
    loss_module = CompositeLoss(task_loss_fn)

    lambdas = {"ais": 1.0, "te": 1.0, "synergy": 1.0}

    # Forward pass
    outputs = model(x)
    # We need to simulate history for AIS/TE
    outputs["prev_encoded"] = torch.randn_like(outputs["encoded"])
    outputs["prev_context_state"] = torch.randn_like(outputs["context_state"])

    loss_dict = loss_module(outputs, y, lambdas)
    loss = loss_dict["loss"]

    # Backward pass
    loss.backward()

    # Check if gradients flowed to the encoder
    assert model.encoder.weight.grad is not None
    # Check if gradients flowed to Storage Nodes
    assert model.storage_nodes[0].recurrent_weights.grad is not None
    # Check if gradients flowed to Synergy Hub
    assert model.synergy_hubs[0].ff_layer.weight.grad is not None

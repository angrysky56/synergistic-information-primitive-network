import pytest
import torch
from sipnet.domain.network.graph import SIPNet

def test_multi_layer_residual():
    hidden_dim = 16
    num_layers = 4
    model = SIPNet(
        input_dim=100,
        hidden_dim=hidden_dim,
        output_dim=10,
        num_layers=num_layers,
        use_embedding=True
    )
    
    x = torch.randint(0, 100, (2, 5))
    outputs = model(x)
    
    assert len(outputs) == 5
    for out in outputs:
        assert out["logits"].shape == (2, 10)
        assert len(out["layer_outputs"]) == num_layers
        assert out["layer_outputs"][0]["final_rep"].shape == (2, hidden_dim)

def test_gradient_flow_multi_layer():
    model = SIPNet(
        input_dim=100,
        hidden_dim=16,
        output_dim=10,
        num_layers=3,
        use_embedding=True
    )
    
    x = torch.randint(0, 100, (2, 3))
    outputs = model(x)
    loss = outputs[-1]["logits"].sum()
    loss.backward()
    
    # Check gradients in all layers
    for i, layer in enumerate(model.layers):
        has_grad = False
        for param in layer.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, f"Layer {i} has no gradients"

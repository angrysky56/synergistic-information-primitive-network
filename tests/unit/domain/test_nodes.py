import torch
import pytest
from sipnet.domain.nodes.storage_node import StorageNode
from sipnet.domain.nodes.transfer_bus import TransferBus
from sipnet.domain.nodes.synergy_hub import SynergyHub

def test_storage_node_dynamics():
    node = StorageNode(dim=10, input_dim=5, spectral_radius=0.9)
    x = torch.randn(1, 5)

    # First forward pass
    state1 = node(x).clone()
    assert state1.shape == (1, 10)
    assert not torch.all(state1 == 0)

    # Passivity test (no input)
    state2 = node(None).clone()
    # Recurrent dynamics should change the state even without input
    assert not torch.all(state1 == state2)

    # Reset
    node.reset_state()
    assert torch.all(node.state == 0)

def test_transfer_bus_routing():
    bus = TransferBus(source_dim=10, target_dim=10)
    x = torch.randn(1, 10)

    output = bus(x)
    assert output.shape == (1, 10)
    assert not torch.all(output == 0)

    # Check activity score update
    bus.set_activity(0.85)
    assert bus.activity_score == 0.85

def test_synergy_hub_integration():
    hub = SynergyHub(ff_dim=10, ctx_dim=10, output_dim=5)
    ff = torch.randn(1, 10)
    ctx = torch.randn(1, 10)

    output = hub(ff, ctx)
    assert output.shape == (1, 5)
    assert not torch.all(output == 0)

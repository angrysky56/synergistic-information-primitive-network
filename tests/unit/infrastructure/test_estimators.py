import torch
import pytest
from sipnet.infrastructure.information_theory.ais_estimator import estimate_mutual_information_renyi, estimate_ais
from sipnet.infrastructure.information_theory.te_estimator import estimate_te
from sipnet.infrastructure.information_theory.pid_estimator import estimate_pid_renyi

def test_mi_renyi_differentiability():
    x = torch.randn(10, 5, requires_grad=True)
    y = torch.randn(10, 5, requires_grad=True)

    mi = estimate_mutual_information_renyi(x, y)
    mi.backward()

    assert x.grad is not None
    assert y.grad is not None
    assert mi >= 0.0

def test_ais_simple():
    # Correlated signals should have AIS > 0
    past = torch.randn(100, 10)
    present = past + torch.randn(100, 10) * 0.1

    ais = estimate_ais(past, present)
    assert ais >= 0.0

    # Uncorrelated signals should have AIS ~ 0
    past_rand = torch.randn(100, 10)
    present_rand = torch.randn(100, 10)
    ais_rand = estimate_ais(past_rand, present_rand)
    assert ais_rand < ais

def test_te_differentiability():
    source_past = torch.randn(10, 5, requires_grad=True)
    target_present = torch.randn(10, 5, requires_grad=True)
    target_past = torch.randn(10, 5, requires_grad=True)

    te = estimate_te(source_past, target_present, target_past)
    te.backward()

    assert source_past.grad is not None
    assert target_present.grad is not None
    assert target_past.grad is not None
    assert te >= 0.0

def test_pid_renyi():
    s1 = torch.randn(100, 5)
    s2 = torch.randn(100, 5)
    # Target is simple sum (Redundant/Synergistic)
    target = s1 + s2 + torch.randn(100, 5) * 0.1

    results = estimate_pid_renyi(s1, s2, target)

    assert results['synergy'] >= 0.0
    assert results['redundancy'] >= 0.0
    assert results['unique1'] >= 0.0
    assert results['unique2'] >= 0.0

    # In a simple additive case, synergy should be high if target
    # provides information not in either source alone
    assert results['synergy'] > 0.0

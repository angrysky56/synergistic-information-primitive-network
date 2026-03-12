import torch
import torch.nn as nn
from sipnet.domain.network.sip_layer import SIPLayer
from sipnet.infrastructure.information_theory.pid_estimator import estimate_pid_renyi

def diagnose():
    print("--- Scientific Diagnosis of SIPLayer Synergy ---")
    torch.manual_seed(42)
    
    # Isolate a single layer
    layer = SIPLayer(input_dim=16, hidden_dim=16, output_dim=16, num_storage_nodes=1, num_synergy_hubs=1, num_parallel_buses=1)
    
    # Simulate a batch of temporal inputs with structured variance
    # We want X_0 to be independent of X_1
    X_0 = torch.randn(100, 16)
    X_1 = torch.randn(100, 16)
    
    # Step 1: Pass X_0
    out_0 = layer.forward_step(X_0)
    
    # Step 2: Pass X_1
    out_1 = layer.forward_step(X_1)
    
    # Check PID at Step 1
    pid_1 = estimate_pid_renyi(out_1["agg_ff_signal"], out_1["agg_ctx_signal"], out_1["final_rep"])
    print("\n--- Step 1 PID ---")
    print(f"Redundancy: {pid_1['redundancy'].item():.4f}")
    print(f"Unique1: {pid_1['unique1'].item():.4f}")
    print(f"Unique2: {pid_1['unique2'].item():.4f}")
    print(f"Synergy: {pid_1['synergy'].item():.4f}")
    
    # Now simulate a forceful backward pass that forces final_rep to be an XOR-like combination
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    
    # Target XOR approximation: sign(X_0) * sign(X_1) is basically XOR in continuous space
    target_logic = torch.sign(X_0) * torch.sign(X_1)
    
    print("\n--- Training to force non-linear dependent output ---")
    for epoch in range(100):
        optimizer.zero_layer = layer.reset_memory()
        
        out_0 = layer.forward_step(X_0)
        out_1 = layer.forward_step(X_1)
        
        loss = nn.MSELoss()(out_1["final_rep"], target_logic)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            pid = estimate_pid_renyi(out_1["agg_ff_signal"], out_1["agg_ctx_signal"], out_1["final_rep"])
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Synergy: {pid['synergy'].item():.4f} | Redundancy: {pid['redundancy'].item():.4f}")

if __name__ == "__main__":
    diagnose()

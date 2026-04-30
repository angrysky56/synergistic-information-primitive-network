import torch
import time
from sipnet.domain.network.graph import SIPNet
from sipnet.application.training.loss_function import CompositeLoss
import torch.nn as nn

def test_vram_scaling():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing VRAM scaling on {device}")
    
    hidden_dims = [16, 32, 64]
    num_layers_list = [2, 4, 8]
    batch_size = 32
    seq_len = 20
    
    for h_dim in hidden_dims:
        for n_layers in num_layers_list:
            print(f"\nConfig: hidden_dim={h_dim}, num_layers={n_layers}")
            try:
                model = SIPNet(
                    input_dim=100,
                    hidden_dim=h_dim,
                    output_dim=10,
                    num_layers=n_layers,
                    use_embedding=True
                ).to(device)
                
                loss_fn = CompositeLoss(nn.CrossEntropyLoss())
                lambdas = {"ais": 1.0, "te": 1.0, "synergy": 1.0, "inter_layer_synergy": 1.0}
                
                x = torch.randint(0, 100, (batch_size, seq_len)).to(device)
                y = torch.randint(0, 10, (batch_size, seq_len)).to(device)
                
                start_time = time.time()
                outputs = model(x)
                
                total_loss = torch.tensor(0.0, device=device)
                for t in range(seq_len):
                    current_out = outputs[t]
                    if t > 0:
                        current_out["prev_layer_outputs"] = outputs[t-1]["layer_outputs"]
                    
                    l_dict = loss_fn(current_out, y[:, t], lambdas)
                    total_loss = total_loss + l_dict["loss"]
                
                total_loss.backward()
                end_time = time.time()
                
                if device == "cuda":
                    vram = torch.cuda.max_memory_allocated() / 1e6
                    print(f"Max VRAM: {vram:.2f} MB | Time: {end_time - start_time:.2f}s")
                    torch.cuda.reset_peak_memory_stats()
                else:
                    print(f"Time: {end_time - start_time:.2f}s")
                
                del model, outputs, total_loss
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Failed: {e}")

if __name__ == "__main__":
    test_vram_scaling()

import torch
from torch.utils.data import Dataset

class DelayedXORDataset(Dataset):
    """
    N-bit Temporal Parity (Delayed XOR) sequence generator.
    Input: [Batch, Seq_Len, N_Bits]
    Targets: [Batch, Seq_Len, N_Bits] (Only the final timestep contains the XOR target)
    """
    def __init__(self, num_samples: int, seq_len: int, num_bits: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_bits = num_bits

        # Generate random bit vectors for t=0 and t=T-1
        self.x0 = torch.randint(0, 2, (num_samples, num_bits), dtype=torch.float32)
        self.xT = torch.randint(0, 2, (num_samples, num_bits), dtype=torch.float32)

        # Target is the element-wise XOR of x0 and xT
        self.y = torch.logical_xor(self.x0, self.xT).float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Initialize empty sequence
        seq = torch.zeros((self.seq_len, self.num_bits), dtype=torch.float32)

        # Inject cues at start and end
        seq[0] = self.x0[idx]
        seq[-1] = self.xT[idx]

        # Initialize targets (all zeros except the final timestep)
        targets = torch.zeros((self.seq_len, self.num_bits), dtype=torch.float32)
        targets[-1] = self.y[idx]

        return seq, targets

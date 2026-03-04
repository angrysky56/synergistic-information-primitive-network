import torch
from torch.utils.data import DataLoader

from ...domain.network.graph import SIPNet
from .loss_function import CompositeLoss


class CognitiveTrainer:
    """
    Trainer for SIP-Net that implements the three phases of cognitive development.
    """

    def __init__(
        self,
        model: SIPNet,
        optimizer: torch.optim.Optimizer,
        loss_module: CompositeLoss,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.device = device

        # Current cognitive phase hyperparameters
        self.lambdas = {"ais": 0.0, "te": 0.0, "synergy": 0.0}
        self.current_phase = 1

    def set_phase(self, phase: int) -> None:
        """Updates lambdas based on the development phase."""
        self.current_phase = phase
        if phase == 1:
            self.lambdas = {"ais": 0.1, "te": 0.1, "synergy": 0.1}
        elif phase == 2:
            self.lambdas = {"ais": 0.1, "te": 1.0, "synergy": 0.1}
        elif phase == 3:
            self.lambdas = {"ais": 1.0, "te": 0.5, "synergy": 1.0}

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        epoch_metrics = {
            "loss": 0.0,
            "task_loss": 0.0,
            "ais": 0.0,
            "te": 0.0,
            "synergy": 0.0,
        }

        for data_seq, target_seq in dataloader:
            data_seq = data_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass unrolls over the entire sequence
            # outputs_seq is a list of dictionaries, one per timestep
            outputs_seq = self.model(data_seq)

            total_loss = torch.tensor(0.0, device=self.device)
            seq_len = len(outputs_seq)

            # Accumulate temporal losses (BPTT)
            for t in range(seq_len):
                current_outputs = outputs_seq[t]

                # Link previous states for Information Theory metrics if t > 0
                if t > 0:
                    current_outputs["prev_encoded"] = outputs_seq[t - 1]["encoded"]
                    current_outputs["prev_context_state"] = outputs_seq[t - 1][
                        "context_state"
                    ]
                    current_outputs["prev_ff_signal"] = outputs_seq[t - 1].get(
                        "ff_signal", outputs_seq[t - 1].get("agg_ff_signal")
                    )
                    current_outputs["prev_final_rep"] = outputs_seq[t - 1]["final_rep"]

                # We only want the task_loss to apply at the final timestep
                # The loss_module handles the targets. We pass the targets for timestep t.
                if target_seq.dim() == 3:
                    current_targets = target_seq[:, t, :]
                else:
                    current_targets = target_seq[:, t]

                loss_dict = self.loss_module(
                    current_outputs, current_targets, self.lambdas
                )

                # For classification tasks with padded intervals, trust ignore_index.
                # For continuous sequences (XOR), zero out intermediate steps unless it's the
                # final target.
                compute_task_loss = True
                if target_seq.dtype != torch.long and t < seq_len - 1:
                    compute_task_loss = False

                if not compute_task_loss:
                    loss_dict["loss"] = loss_dict["loss"] - loss_dict["task_loss"]
                    loss_dict["task_loss"] = torch.tensor(0.0, device=self.device)

                total_loss = total_loss + loss_dict["loss"]

                total_loss = total_loss + loss_dict["loss"]

                # Dynamically accumulate metrics for logging
                for k, v in loss_dict.items():
                    # Handle single scalar tensors
                    if isinstance(v, torch.Tensor) and v.dim() == 0:
                        if k not in epoch_metrics:
                            epoch_metrics[k] = 0.0
                        epoch_metrics[k] += v.item() / seq_len
                    # Handle lists of tensors (like te_buses)
                    elif isinstance(v, list):
                        for idx, v_item in enumerate(v):
                            list_k = f"{k}_{idx}"
                            if list_k not in epoch_metrics:
                                epoch_metrics[list_k] = 0.0
                            epoch_metrics[list_k] += v_item.item() / seq_len

            # Single backward pass through the entire unrolled sequence
            total_loss.backward()
            self.optimizer.step()

        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)

        return epoch_metrics

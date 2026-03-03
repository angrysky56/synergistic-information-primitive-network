
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
        device: str = "cpu"
    ):
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
            "loss": 0.0, "task_loss": 0.0, "ais": 0.0, "te": 0.0, "synergy": 0.0
        }

        prev_encoded = None
        prev_context = None

        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(data)

            if prev_encoded is not None:
                outputs["prev_encoded"] = prev_encoded.detach()
            if prev_context is not None:
                outputs["prev_context_state"] = prev_context.detach()

            prev_encoded = outputs["encoded"]
            prev_context = outputs["context_state"]

            loss_dict = self.loss_module(outputs, targets, self.lambdas)
            loss_dict["loss"].backward()
            self.optimizer.step()

            for k in epoch_metrics:
                if k in loss_dict:
                    epoch_metrics[k] += loss_dict[k].item()

        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)

        return epoch_metrics

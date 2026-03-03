
import torch
import torch.nn as nn

from ...infrastructure.information_theory.ais_estimator import estimate_ais
from ...infrastructure.information_theory.pid_estimator import estimate_pid_renyi
from ...infrastructure.information_theory.te_estimator import estimate_te


class CompositeLoss(nn.Module):
    """
    Multi-objective loss function for SIP-Net.
    L = L_task - lambda1*AIS - lambda2*TE - lambda3*Synergy
    """
    def __init__(self, task_loss_fn: nn.Module):
        super().__init__()
        self.task_loss_fn = task_loss_fn

        # State tracking for TE (requires past values)
        self.prev_source: dict[str, torch.Tensor] = {}
        self.prev_target: dict[str, torch.Tensor] = {}

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        lambdas: dict[str, float]
    ) -> dict[str, torch.Tensor]:
        """
        Calculates combined loss and individual metrics.
        """
        # 1. Task Loss
        l_task = self.task_loss_fn(outputs["logits"], targets)

        # In sequential contexts with padded targets, loss functions like CrossEntropy
        # may return NaN if all targets in the batch are ignored at this timestep.
        if torch.isnan(l_task):
            # We preserve gradient flow compatibility by mimicking a zero-loss state
            l_task = torch.tensor(0.0, device=l_task.device)

        # 2. AIS Reward (Calculated on Storage Nodes)
        # Assuming we have saved state from t-1 in outputs or internal buffer
        # For simplicity in this implementation, we compare current context to previous context
        # (Alternatively, compare state to its own past if sequence is preserved)
        ais_val = torch.tensor(0.0, device=l_task.device)
        if "prev_context_state" in outputs:
            # Check for batch size mismatch (e.g. final batch of epoch)
            if outputs["prev_context_state"].shape[0] == outputs["context_state"].shape[0]:
                ais_val = estimate_ais(outputs["prev_context_state"], outputs["context_state"])

        # 3. TE Reward (Transfer Buses)
        te_val = torch.tensor(0.0, device=l_task.device)
        # Tracking true TE mathematically: Info transitting from past memory into current decision outputs.
        if "prev_context_state" in outputs and "prev_final_rep" in outputs:
            # Check for batch size mismatch
            if outputs["prev_context_state"].shape[0] == outputs["final_rep"].shape[0]:
                te_val = estimate_te(
                    source_past=outputs["prev_context_state"],
                    target_present=outputs["final_rep"],
                    target_past=outputs["prev_final_rep"]
                )

        # 4. Synergy Reward (Synergy Hubs)
        # Synergy(Encoding, Context -> Final Representation)
        pid_results = estimate_pid_renyi(
            s1=outputs["ff_signal"],
            s2=outputs["ctx_signal"],
            target=outputs["final_rep"]
        )
        synergy_val = pid_results["synergy"]

        # 5. Combined Loss
        # We maximize rewards by subtracting them
        total_loss = (
            l_task
            - lambdas.get("ais", 0.0) * ais_val
            - lambdas.get("te", 0.0) * te_val
            - lambdas.get("synergy", 0.0) * synergy_val
        )

        return {
            "loss": total_loss,
            "task_loss": l_task,
            "ais": ais_val,
            "te": te_val,
            "synergy": synergy_val,
            "redundancy": pid_results["redundancy"]
        }

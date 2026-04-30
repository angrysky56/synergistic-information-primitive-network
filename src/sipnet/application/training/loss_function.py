"""
Loss functions for SIP-Net training.
"""

import torch
import torch.nn as nn

from ...domain.common.types import StepOutput
from ...infrastructure.information_theory.ais_estimator import estimate_ais
from ...infrastructure.information_theory.pid_estimator import estimate_pid_renyi
from ...infrastructure.information_theory.te_estimator import estimate_te


class CompositeLoss(nn.Module):
    """
    Multi-objective loss function for SIP-Net.
    L = L_task - lambda1*AIS - lambda2*TE - lambda3*Synergy
    """

    def __init__(self, task_loss_fn: nn.Module) -> None:
        super().__init__()
        self.task_loss_fn = task_loss_fn

        # State tracking for TE (requires past values)
        self.prev_source: dict[str, torch.Tensor] = {}
        self.prev_target: dict[str, torch.Tensor] = {}

    def forward(
        self,
        outputs: StepOutput,
        targets: torch.Tensor,
        lambdas: dict[str, float],
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

        total_ais = torch.tensor(0.0, device=l_task.device)
        total_te = torch.tensor(0.0, device=l_task.device)
        total_synergy = torch.tensor(0.0, device=l_task.device)
        total_l1 = torch.tensor(0.0, device=l_task.device)
        total_redundancy = torch.tensor(0.0, device=l_task.device)
        total_cross_bus_red = torch.tensor(0.0, device=l_task.device)

        num_layers = len(outputs["layer_outputs"])

        # Iterate through every SIPLayer to calculate information flow
        for l_idx in range(num_layers):
            layer_out = outputs["layer_outputs"][l_idx]

            # 1. AIS
            if outputs.get("prev_layer_outputs") is not None:
                prev_layer_outputs = outputs["prev_layer_outputs"]
                if prev_layer_outputs is not None:
                    prev_layer_out = prev_layer_outputs[l_idx]
                    if (
                        prev_layer_out["context_state"].shape[0]
                        == layer_out["context_state"].shape[0]
                    ):
                        total_ais = total_ais + estimate_ais(
                            prev_layer_out["context_state"], layer_out["context_state"]
                        )

            # 2. TE & L1 Cost
            if outputs.get("prev_layer_outputs") is not None:
                prev_layer_outputs = outputs["prev_layer_outputs"]
                if prev_layer_outputs is not None:
                    prev_layer_out = prev_layer_outputs[l_idx]
                    if (
                        prev_layer_out["context_state"].shape[0]
                        == layer_out["final_rep"].shape[0]
                    ):
                        for bus_output in layer_out["ctx_signals"]:
                            # Just track L1 cost
                            total_l1 = total_l1 + torch.abs(bus_output).mean()

                        total_te = total_te + estimate_te(
                            source_past=prev_layer_out["context_state"],
                            target_present=layer_out["agg_ctx_signal"],
                            target_past=prev_layer_out["final_rep"],
                        )
            elif "ctx_signals" in layer_out:
                for bus_output in layer_out["ctx_signals"]:
                    total_l1 = total_l1 + torch.abs(bus_output).mean()

            # 3. Synergy & Redundancy
            # Synergy checks mutual info across FF and TD bounds relative to output
            pid_results = estimate_pid_renyi(
                s1=layer_out["agg_ff_signal"],
                s2=layer_out["agg_ctx_signal"],
                target=layer_out["final_rep"],
            )
            total_synergy = total_synergy + pid_results["synergy"]
            total_redundancy = total_redundancy + pid_results["redundancy"]

            if len(layer_out["ctx_signals"]) > 1:
                layer_cross_red = torch.tensor(0.0, device=l_task.device)
                for i in range(len(layer_out["ctx_signals"]) - 1):
                    pid_bus = estimate_pid_renyi(
                        s1=layer_out["ctx_signals"][i],
                        s2=layer_out["ctx_signals"][i + 1],
                        target=layer_out["final_rep"],
                    )
                    layer_cross_red = layer_cross_red + pid_bus["redundancy"]
                total_cross_bus_red = total_cross_bus_red + (
                    layer_cross_red / (len(layer_out["ctx_signals"]) - 1)
                )

        total_inter_layer_syn = torch.tensor(0.0, device=l_task.device)
        # 4. Inter-Layer Synergy
        for i in range(num_layers - 1):
            l1_out = outputs["layer_outputs"][i]
            l2_out = outputs["layer_outputs"][i+1]
            
            # Ensure shape compatibility for PID (assuming same hidden_dim)
            inter_pid = estimate_pid_renyi(
                s1=l1_out["final_rep"],
                s2=l2_out["final_rep"],
                target=l2_out["context_state"], # Novelty relative to next layer's state
            )
            total_inter_layer_syn = total_inter_layer_syn + inter_pid["synergy"]

        # Average metrics across layers for stable scaling
        if num_layers > 0:
            total_ais = total_ais / num_layers
            total_te = total_te / num_layers
            total_synergy = total_synergy / num_layers
            total_l1 = total_l1 / num_layers
            total_cross_bus_red = total_cross_bus_red / num_layers
            total_redundancy = total_redundancy / num_layers
            if num_layers > 1:
                total_inter_layer_syn = total_inter_layer_syn / (num_layers - 1)

        lambda_te = lambdas.get("te", 0.0)
        total_loss = (
            l_task
            - lambdas.get("ais", 0.0) * total_ais
            - lambda_te * total_te
            - lambdas.get("synergy", 0.0) * total_synergy
            - lambdas.get("inter_layer_synergy", 0.0) * total_inter_layer_syn
            + (lambda_te * 0.1) * total_l1
            + (lambda_te * 2.0) * total_cross_bus_red
        )

        return {
            "loss": total_loss,
            "task_loss": l_task,
            "ais": total_ais,
            "te": total_te,
            "synergy": total_synergy,
            "inter_layer_syn": total_inter_layer_syn,
            "redundancy": total_redundancy,
            "l1_cost": total_l1,
            "cross_bus_red": total_cross_bus_red,
            # Placeholder for backwards compatibility reporting
            "te_buses": [],
        }

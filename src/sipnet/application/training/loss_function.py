"""
Loss functions for SIP-Net training.
"""

import torch
import torch.nn as nn

from ...infrastructure.information_theory.ais_estimator import estimate_ais
from ...infrastructure.information_theory.pid_estimator import estimate_pid_renyi
from ...infrastructure.information_theory.te_estimator import estimate_te


class CompositeLoss(nn.Module):  # type: ignore
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
        outputs: dict[str, torch.Tensor],
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

        # 2. AIS Reward (Calculated on Storage Nodes)
        # Assuming we have saved state from t-1 in outputs or internal buffer
        # For simplicity in this implementation, we compare current context to previous context
        # (Alternatively, compare state to its own past if sequence is preserved)
        ais_val = torch.tensor(0.0, device=l_task.device)
        if "prev_context_state" in outputs:
            # Check for batch size mismatch (e.g. final batch of epoch)
            if (
                outputs["prev_context_state"].shape[0]
                == outputs["context_state"].shape[0]
            ):
                ais_val = estimate_ais(
                    outputs["prev_context_state"], outputs["context_state"]
                )

        # 3. TE Reward (Transfer Buses)
        te_val = torch.tensor(0.0, device=l_task.device)
        l1_penalty = torch.tensor(0.0, device=l_task.device)

        # Tracking true TE mathematically across all active buses
        if (
            "prev_context_state" in outputs
            and "prev_final_rep" in outputs
            and "ctx_signals" in outputs
        ):
            # Check for batch size mismatch
            if outputs["prev_context_state"].shape[0] == outputs["final_rep"].shape[0]:
                te_vals = []
                for bus_output in outputs["ctx_signals"]:
                    te_vals.append(
                        estimate_te(
                            source_past=outputs["prev_context_state"],
                            target_present=bus_output,  # Evaluate TE across bus output
                            target_past=outputs["prev_final_rep"],
                        )
                    )

                    # Apply L1 structural penalty to the bus signal itself (metabolic cost)
                    l1_penalty = l1_penalty + torch.abs(bus_output).mean()

                # Evaluate TE reward strictly on the aggregated output.
                # Avoid summing individual TEs to prevent native scaling incentives.
                if "agg_ctx_signal" in outputs:
                    te_val = estimate_te(
                        source_past=outputs["prev_context_state"],
                        target_present=outputs["agg_ctx_signal"],
                        target_past=outputs["prev_final_rep"],
                    )
                elif te_vals:
                    te_val = torch.stack(
                        te_vals
                    ).max()  # Fallback: penalize duplicating rewards

        # 4. Synergy Reward (Synergy Hubs)
        # Synergy(Encoding, Context -> Final Representation)
        s1_sig = outputs.get("agg_ff_signal", outputs.get("ff_signal"))
        if s1_sig is None:
            # Fallback for type safety, should be present in correct flow
            s1_sig = torch.zeros_like(outputs["final_rep"])

        s2_sig = outputs.get("agg_ctx_signal")
        if s2_sig is None:
            s2_sig = (
                outputs["ctx_signals"][0]
                if "ctx_signals" in outputs and outputs["ctx_signals"]
                else outputs["ctx_signal"]
            )

        pid_results = estimate_pid_renyi(
            s1=s1_sig,
            s2=s2_sig,
            target=outputs["final_rep"],
        )
        synergy_val = pid_results["synergy"]
        redundancy_pen = pid_results["redundancy"]

        cross_bus_redundancy = torch.tensor(0.0, device=l_task.device)
        if "ctx_signals" in outputs and len(outputs["ctx_signals"]) > 1:
            for i in range(len(outputs["ctx_signals"]) - 1):
                pid_bus = estimate_pid_renyi(
                    s1=outputs["ctx_signals"][i],
                    s2=outputs["ctx_signals"][i + 1],
                    target=outputs["final_rep"],
                )
                cross_bus_redundancy = cross_bus_redundancy + pid_bus["redundancy"]

            cross_bus_redundancy = cross_bus_redundancy / (
                len(outputs["ctx_signals"]) - 1
            )

        # 5. Combined Loss
        # We maximize rewards by subtracting them. We minimize penalties by adding them.
        # Here we add the L1 constraints and Redundancy metrics scaling against the TE rewards
        lambda_te = lambdas.get("te", 0.0)

        total_loss = (
            l_task
            - lambdas.get("ais", 0.0) * ais_val
            - lambda_te * te_val
            - lambdas.get("synergy", 0.0) * synergy_val
            + (lambda_te * 0.1) * l1_penalty  # Metabolic cost of firing
            + (lambda_te * 2.0)
            * cross_bus_redundancy  # Penalty for identical structural paths!
        )

        return {
            "loss": total_loss,
            "task_loss": l_task,
            "ais": ais_val,
            "te": te_val,
            "te_buses": te_vals if "te_vals" in locals() and te_vals else [],
            "synergy": synergy_val,
            "redundancy": redundancy_pen,
            "l1_cost": l1_penalty,
            "cross_bus_red": cross_bus_redundancy,
        }

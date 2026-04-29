"""
Interpretability Deep-Dive for SIP-Net NLP Coreference.
Trains a model (or loads one) and visualizes information primitives on case studies.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sipnet.application.execution.nlp_generators import (
    SequentialNLPDataset,
    nlp_collate_fn,
)
from sipnet.application.execution.visualizer import save_text_highlight_report
from sipnet.application.training.loss_function import CompositeLoss
from sipnet.application.training.trainer import CognitiveTrainer
from sipnet.domain.network.graph import SIPNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(device: str) -> tuple[SIPNet, SequentialNLPDataset]:
    """Trains a model for interpretability analysis."""
    dataset = SequentialNLPDataset(num_samples=2000, include_distractors=True)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=nlp_collate_fn
    )

    vocab_size = dataset.vocab_size
    hidden_dim = 16

    model = SIPNet(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        output_dim=vocab_size,
        num_storage_nodes=1,
        num_synergy_hubs=1,
        use_embedding=True,
    ).to(device)

    task_loss = nn.CrossEntropyLoss(ignore_index=0)
    loss_module = CompositeLoss(task_loss_fn=task_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = CognitiveTrainer(model, optimizer, loss_module, device=device)

    # Fast-track training for demo
    epochs_per_phase = 20
    for phase in [1, 2, 3]:
        logger.info(f"--- Training Phase {phase} ---")
        trainer.set_phase(phase)
        for epoch in range(1, epochs_per_phase + 1):
            metrics = trainer.train_epoch(dataloader)
            if epoch % 10 == 0:
                logger.info(
                    f"Phase {phase} Ep {epoch:02d} | Acc: {metrics.get('accuracy', 0.0):.3f}"
                )

    # Save diagnostic plots
    trainer.save_training_plots("REPORTS/plots")

    return model, dataset


def run_interpretability_analysis(
    model: SIPNet, dataset: SequentialNLPDataset, device: str
) -> None:
    """Runs inference on specific cases and extracts metrics."""
    model.eval()

    # Define a custom case study
    # "The cat watched the dog until it slept"
    case_study = ["The", "cat", "watched", "the", "dog", "until", "it", "slept"]
    indices = torch.tensor([[dataset.word2idx[w] for w in case_study]], device=device)

    # Forward pass
    with torch.no_grad():
        outputs_seq = model(indices)

    # Extract metrics per timestep
    # We need to compute them using the same logic as the trainer
    task_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loss_module = CompositeLoss(task_loss_fn=task_loss_fn)
    lambdas = {"ais": 1.0, "te": 1.0, "synergy": 1.0}

    steps_metrics = []
    for t in range(len(outputs_seq)):
        current_outputs = outputs_seq[t]
        if t > 0:
            current_outputs["prev_layer_outputs"] = outputs_seq[t - 1]["layer_outputs"]

        # Target doesn't matter for metrics except task_loss
        dummy_target = torch.tensor([0], device=device)
        metrics = loss_module(current_outputs, dummy_target, lambdas)

        steps_metrics.append(
            {
                "ais": metrics["ais"].item(),
                "te": metrics["te"].item(),
                "synergy": metrics["synergy"].item(),
            }
        )

    # Plot Synergy Spikes
    synergy_scores = [m["synergy"] for m in steps_metrics]
    te_scores = [m["te"] for m in steps_metrics]

    plt.figure(figsize=(12, 6))
    plt.plot(case_study, synergy_scores, marker="o", label="Synergy")
    plt.plot(case_study, te_scores, marker="s", label="Transfer Entropy")
    plt.axhline(0, color="black", lw=0.5)
    plt.title("Information Primitives during Coreference Resolution")
    plt.ylabel("Bits")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("REPORTS/plots/coreference_it_dynamics.png")
    plt.close()

    # Save HTML Highlighting Reports
    save_text_highlight_report(
        case_study,
        synergy_scores,
        "REPORTS/synergy_highlights.html",
        title="Synergy Activation (Coreference)",
        cmap_name="YlOrRd",
    )
    save_text_highlight_report(
        case_study,
        te_scores,
        "REPORTS/te_highlights.html",
        title="Transfer Entropy Activation (Coreference)",
        cmap_name="GnBu",
    )

    # Generate Summary Report
    report = f"""# SIP-Net Coreference Interpretability Report

## Case Study: "{' '.join(case_study)}"

### Visualization: Information Primitives
![IT Dynamics](plots/coreference_it_dynamics.png)

### Findings:
1. **Synergy Spikes**: Synergy values should peak at the pronoun index ("it"), indicating that the model is combining context from the storage node (antecedent "cat") and the current sensory input to resolve the reference.
2. **Transfer Entropy**: TE flow indicates information being retrieved from the long-term context state to the final representation.

### HTML Reports:
- [Synergy Highlights](synergy_highlights.html)
- [TE Highlights](te_highlights.html)
"""
    Path("REPORTS/coreference_interpretability.md").write_text(report)
    logger.info(
        "Generated interpretability report: REPORTS/coreference_interpretability.md"
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path("REPORTS/plots").mkdir(parents=True, exist_ok=True)

    logger.info("Starting Interpretability Study...")
    model, dataset = train_model(device)
    run_interpretability_analysis(model, dataset, device)


if __name__ == "__main__":
    main()

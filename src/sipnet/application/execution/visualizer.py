"""Visualization suite for SIP-Net information primitives."""

import html
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_metric_matrix(
    matrix: np.ndarray | list[list[float]],
    title: str,
    path: str | Path,
    cmap: str = "rocket",
    annot: bool = False,
) -> None:
    """
    Plot a Source -> Destination heatmap of an information metric.

    Args:
        matrix: The square matrix to plot (N x N).
        title: Title of the plot.
        path: File path to save the plot.
        cmap: Color map to use.
        annot: Whether to annotate the heatmap with values.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap=cmap, annot=annot, fmt=".2f", cbar_kws={"label": "Value"})
    plt.title(title)
    plt.xlabel("Destination IPP Index")
    plt.ylabel("Source IPP Index")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved metric matrix plot to {path}")


def plot_training_dynamics(
    metrics_log: list[dict[str, float]],
    path: str | Path,
    title: str = "SIP-Net Training Dynamics",
) -> None:
    """
    Plot training metrics over time.

    Args:
        metrics_log: A list of dictionaries containing metric values per step/epoch.
        path: File path to save the plot.
        title: Title of the plot.
    """
    if not metrics_log:
        logger.warning("No metrics to plot training dynamics.")
        return

    df = pd.DataFrame(metrics_log)
    steps = range(len(df))

    # Identify metric types
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    acc_cols = [c for c in df.columns if "accuracy" in c.lower()]
    ipp_cols = [
        c for c in df.columns if any(ipp in c.upper() for ipp in ["AIS", "TE", "SYN"])
    ]

    num_plots = sum([1 if loss_cols else 0, 1 if acc_cols else 0, 1 if ipp_cols else 0])
    if num_plots == 0:
        logger.warning("No recognizable columns in metrics_log to plot.")
        return

    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    ax_idx = 0

    # Plot Loss
    if loss_cols:
        for col in loss_cols:
            axes[ax_idx].plot(steps, df[col], label=col)
        axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].legend()
        axes[ax_idx].set_title("Loss Dynamics")
        ax_idx += 1

    # Plot Accuracy
    if acc_cols:
        for col in acc_cols:
            axes[ax_idx].plot(steps, df[col], label=col)
        axes[ax_idx].set_ylabel("Accuracy")
        axes[ax_idx].legend()
        axes[ax_idx].set_title("Accuracy Dynamics")
        ax_idx += 1

    # Plot Information Primitives
    if ipp_cols:
        for col in ipp_cols:
            axes[ax_idx].plot(steps, df[col], label=col)
        axes[ax_idx].set_ylabel("IPP Value (Bits)")
        axes[ax_idx].legend()
        axes[ax_idx].set_title("Information Primitive Dynamics")
        ax_idx += 1

    plt.xlabel("Step / Epoch")
    plt.suptitle(title)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved training dynamics plot to {path}")


def save_text_highlight_report(
    tokens: list[str],
    scores: list[float] | np.ndarray,
    path: str | Path,
    title: str = "Text Activation Report",
    cmap_name: str = "YlOrRd",
) -> None:
    """
    Generate an HTML report highlighting tokens based on scores.

    Args:
        tokens: List of tokens.
        scores: Activation or IT scores for each token.
        path: File path to save the HTML report.
        title: Title of the report.
        cmap_name: Name of the matplotlib colormap to use.
    """
    if len(tokens) != len(scores):
        logger.error(f"Tokens length ({len(tokens)}) != Scores length ({len(scores)})")
        return

    # Normalize scores for coloring
    scores_arr = np.array(scores)
    s_min, s_max = scores_arr.min(), scores_arr.max()
    if s_max > s_min:
        norm_scores = (scores_arr - s_min) / (s_max - s_min)
    else:
        norm_scores = np.zeros_like(scores_arr)

    cmap = plt.get_cmap(cmap_name)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: 'Inter', -apple-system, sans-serif; line-height: 1.6; padding: 40px; background: #f8f9fa; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .token-container {{ display: flex; flex-wrap: wrap; gap: 4px; font-size: 1.1em; }}
            .token {{ padding: 2px 4px; border-radius: 4px; transition: transform 0.2s; cursor: default; }}
            .token:hover {{ transform: scale(1.1); box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
            .legend {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <div class="token-container">
    """

    for token, score, norm_score in zip(tokens, scores, norm_scores):
        rgba = cmap(norm_score)
        color = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.4)"
        safe_token = html.escape(token)
        html_content += f'            <span class="token" style="background-color: {color};" title="Score: {score:.4f}">{safe_token}</span>\n'

    html_content += """
            </div>
            <div class="legend">
                <p>Higher intensity indicates higher information synergy/transfer.</p>
            </div>
        </div>
    </body>
    </html>
    """

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Saved text highlight report to {path}")

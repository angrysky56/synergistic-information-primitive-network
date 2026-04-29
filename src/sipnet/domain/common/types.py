from typing import TypedDict, List, Optional
import torch

class LayerOutput(TypedDict):
    """Output dictionary for a single SIPLayer step."""
    encoded: torch.Tensor
    ff_signals: List[torch.Tensor]
    ctx_signals: List[torch.Tensor]
    agg_ff_signal: torch.Tensor
    agg_ctx_signal: torch.Tensor
    context_state: torch.Tensor
    final_rep: torch.Tensor

class StepOutput(TypedDict):
    """Output dictionary for a single SIPNet network step."""
    logits: torch.Tensor
    layer_outputs: List[LayerOutput]
    # Optional field injected during training to link t and t-1
    prev_layer_outputs: Optional[List[LayerOutput]]

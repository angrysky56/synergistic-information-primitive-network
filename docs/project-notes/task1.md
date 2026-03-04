# SIP-Net Future Plans: Cognitive Validation

Now that our infrastructure utilizes robust Matrix-based Rényi's $\alpha$-Entropy estimators, the next step is applying SIP-Net to a dataset complex enough to demand these specialized information primitives.

## Phase 1: Temporal Environment Design (Planning) (Completed)
- [x] Define a sequential benchmark task (N-bit Temporal Parity / Delayed XOR).
    - *Goal*: The task must require **temporally delayed logic** (to force AIS) and **conditional feature routing** (to force TE and Synergy).
- [x] Evaluate the current [CognitiveTrainer](file:///home/ty/Repositories/ai_workspace/synergistic-information-primitive-network/src/sipnet/application/training/trainer.py#9-115) loop. Ensure cross-batch sequence memory is correctly preserved via `detach()` for true BPTT/AIS across long time horizons.

## Phase 2: Implementation (Execution) (Completed)
- [x] Create the new data generator/loader for Delayed XOR.
- [x] Refactor the execution loop in `SIPNet.forward` and [CognitiveTrainer](file:///home/ty/Repositories/ai_workspace/synergistic-information-primitive-network/src/sipnet/application/training/trainer.py#9-115) to properly manage temporal hidden states across sequential steps (instead of single-step contexts).

## Phase 3: Visualization & Extreme Interpretability (Verification) (Completed)
- [x] Build a diagnostic script to extract and visualize the $\lambda$ coefficients against the $L_{task}$, $AIS$, $TE$, and $Synergy$ metrics over time.
- [x] Output a dependency graph showing which TransferBuses were "pruned" (TE $\to 0$) versus which became dominant highways.

## Phase B: Sequential NLP Coreference (Completed)
- [x] Draft [SequentialNLPDataset](file:///home/ty/Repositories/ai_workspace/synergistic-information-primitive-network/src/sipnet/application/execution/nlp_generators.py#5-66) generator to construct sequences (e.g. `["The", "dog", "barked", "until", "it", "slept"]`) to force non-linear TE coreference associations.
- [x] Integrate text-token embedding layers into [SIPNet](file:///home/ty/Repositories/ai_workspace/synergistic-information-primitive-network/src/sipnet/domain/network/graph.py#10-143) to translate discrete syntax into continuous manifolds prior to TE tracking.

## Phase C: Structural Scaling & Dynamic Routing (Active)
- [x] Refactor [SIPNet](file:///home/ty/Repositories/ai_workspace/synergistic-information-primitive-network/src/sipnet/domain/network/graph.py#10-143) to support a parameterizable `ModuleList` of generic [TransferBus](file:///home/ty/Repositories/ai_workspace/synergistic-information-primitive-network/src/sipnet/domain/nodes/transfer_bus.py#5-31) lines running in parallel to simulate routing redundancy options.
- [x] Apply **L1 Regularization** to the specific active [TransferBus](file:///home/ty/Repositories/ai_workspace/synergistic-information-primitive-network/src/sipnet/domain/nodes/transfer_bus.py#5-31) signal vectors output parameters penalizing zero-impact logic natively inside [CompositeLoss](file:///home/ty/Repositories/ai_workspace/synergistic-information-primitive-network/src/sipnet/application/training/loss_function.py#10-108).
- [x] Tie Synergy's Matrix-Rényi PID *Redundancy* tracking variable to the global [CompositeLoss](file:///home/ty/Repositories/ai_workspace/synergistic-information-primitive-network/src/sipnet/application/training/loss_function.py#10-108) ensuring multi-bus identical streams incur intrinsic metabolic limits and prune towards single pathways natively.


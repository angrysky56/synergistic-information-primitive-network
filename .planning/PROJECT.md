# SIP-Net: Synergistic Information Primitive Network

## What This Is
A novel artificial neural network architecture inspired by biological information processing principles. SIP-Net utilizes a heterogeneous graph of specialized Information Processing Primitives (IPPs)—Storage, Transfer, and Synergy—regularized via Matrix-based Rényi entropy metrics (AIS, TE, Synergy) to achieve high interpretability and functional specialization.

## Core Value
To bridge the gap between abstract information theory and differentiable neural architectures, creating networks that are biologically plausible, mathematically grounded, and extremely transparent in their internal logic.

## Context
The project has moved from theoretical design to core implementation (domain, infrastructure, application layers). Codebase mapping is complete. The architecture uses Domain-Driven Design (DDD) and Clean Architecture.

## Requirements

### Validated
- ✓ **Clean Architecture Foundation**: Core directory structure (Domain, Infrastructure, Application, Interfaces).
- ✓ **IPP Prototypes**: Initial implementation of `StorageNode`, `TransferBus`, and `SynergyHub`.
- ✓ **Matrix-Rényi Estimators**: Differentiable kernel-based estimators for AIS, TE, and PID.
- ✓ **Training Pipeline**: Cognitive phasing scheduler (Redundant -> Pruning -> Specialization).

### Active
- [ ] **Architecture Hardening**: Achieve 100% `mypy` and `pytest` compliance.
- [ ] **Stability Verification**: Eliminate runtime instabilities (NaNs, exploding gradients) in the training loop.
- [ ] **Technical Debt Documentation**: Identify and document non-obvious fixes or architectural gaps discovered during hardening.
- [ ] **NLP Evaluation**: Train and evaluate on a complex semantic coreference task (deferred until hardening is complete).

### Out of Scope
- [ ] **Scaling to Large LLMs**: Focused on primitive specialization and interpretability on abstract tasks first.
- [ ] **Real-time API**: Deployment interfaces are secondary to mathematical validation.

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Matrix-based Rényi Entropy | Bypasses expensive eigendecompositions while capturing non-linear dependencies in O(N²) time. | Validated |
| Cognitive Phasing | Orchestrates functional specialization over three distinct phases to mimic biological development. | In Progress |
| Domain-Driven Design | Ensures high maintainability and testability for a complex, heterogeneous architecture. | Validated |

## Evolution
This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone**:
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-29 after GSD initialization*

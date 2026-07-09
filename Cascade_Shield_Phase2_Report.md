# Cascade Shield: Final Project Report

## Executive Summary
Cascade Shield is an infrastructure cascade failure prediction system using Graph Neural Networks (GNNs) to predict how anomalies propagate through complex topological networks. The project built a 4-layer heterogeneous graph (Power/Water/Road/Transit), a GNN-driven Monte Carlo SEIR hybrid engine, and a SCADA-styled React dashboard for live scenario visualization.

This report documents **two fully verified tri-layer dataset training runs**, both conducted at the 1000m topological fusion threshold:
1. **Phase 1 Baseline (Deprecated):** Achieved **0.9004 ± 0.0010 ROC-AUC**, but was generated with a bug in the traffic-parser (weight pollution), faulty topology serialization, and unconstrained random cascade origins.
2. **Phase 2 ICS-Driven (Current Headline):** After remediating the parser and topology bugs, the engine was driven strictly by telemetry from 4 actual Industrial Control System (ICS) power nodes. The final, verified architecture achieves **0.9089 ± 0.0003 ROC-AUC** and **0.6790 ± 0.0060 P@K** on this dataset.

All numerical claims in this report are backed by explicit script execution and saved N=3 checkpoint metadata. Any historical claims that could not be traced to a literal output have been removed.

---

## 1. Experimental Journey & Loss Re-engineering

### The "Timidity" Hypothesis
Initial evaluations of `CascadeNet` utilizing Focal Loss revealed a degenerate behavior: the model was hedging. Because cascading failures are extremely rare (91.8% of nodes remain safe), the model learned to aggressively suppress its predictions across the board to avoid massive penalties.
- This resulted in an artificially inflated `Containment_Accuracy` of `1.0`, as the model rarely predicted any node above the `0.5` threshold. We ultimately concluded that `Containment_Accuracy` was a degenerate metric under class-imbalance-induced score suppression and excluded it from serious reporting.
- The Time-to-Impact Mean Absolute Error (MAE) was similarly affected, though it was eventually corrected to `4.39` via a `Softplus` activation fix.

### Transition to Pairwise Ranking Loss
To break the model out of its class-imbalance timidity, we shifted the paradigm from absolute classification to **Pairwise Ranking Loss**. By rewarding the model for correctly ranking infected nodes higher than safe nodes *within the same graph*, the model was forced to learn the relative risk of propagation without being penalized for absolute scale.

> [!NOTE]
> **Unverified Early Pipeline:** Sections 1–3 of earlier report versions cited a "0.735 ± 0.033 ROC-AUC ceiling" measured on a 240-node dense synthetic graph, alongside numbers such as "0.9576 SEIR AUC" and "0.9480 Naive Distance AUC." A global search of the repository found **no training log, checkpoint metadata, or script output file** containing these numbers. They have been removed from this report. The experimental narrative above (timidity, ranking loss, winner's curse) describes genuine architectural work and code that exists in the repository, but the specific AUC figures that appeared alongside it could not be verified.

---

## 2. Establishing the Naive Distance Baseline

To properly contextualize the GNN's performance, we developed a **Naive Distance Heuristic** that ranks every node by its topological hop-distance from the cascade origin. Given the deterministic nature of the SEIR transmission simulation, closer nodes are more likely to be infected. 

On the verified **Phase 2 test split**, the naive heuristic scores **0.8991 ROC-AUC** and **0.6601 P@K**, providing the key competitive upper bound for the final architecture.

The GNN barely edges out this simple Dijkstra heuristic (0.9089 vs. 0.8991 AUC). This suggests the GNN is primarily learning to approximate topological distance rather than discovering novel cascade dynamics.

---

## 3. The Val-Test Gap

On the verified Phase 2 training run (N=3 seeds), the Val-Test gap was essentially zero:
- **Phase 2 Seed 0:** Val 0.9044 vs Test 0.9084
- **Phase 2 Seed 1:** Val 0.9064 vs Test 0.9091
- **Phase 2 Seed 2:** Val 0.9048 vs Test 0.9091

This functionally zero gap is explained by the strict 4-origin ICS constraint (detailed in Section 5.2): the model only ever trains and tests on cascade footprints originating from the same four grid points. Because the test distribution is not just macroscopically but *microscopically* identical to the validation distribution, the model suffers virtually no "winner's curse".

---

## 4. The Interactive Dashboard

To make these findings tangible, we designed and implemented a control-room-styled dashboard integrated with a React/Vite frontend and FastAPI backend.

![Cascade Shield Interactive Dashboard - Main View](C:/Users/SPOORTHI/.gemini/antigravity/brain/867cd415-9ae4-4d30-ad0b-42eee080c434/media__1783071232056.png)

### SCADA-Instrument Aesthetic
We eschewed generic analytics styling in favor of a "control-room" aesthetic appropriate for critical infrastructure monitoring:
- Deep blueprint-navy background with a cyan grid overlay.
- `JetBrains Mono` and `Inter` typography for an instrument-panel feel.
- A custom HTML5 Canvas `react-force-graph-2d` renderer featuring pulsing origin beacons and a thermal color scale (slate → teal → amber → red) to represent topological load rather than generic heatmaps.

### Live Metrics & Score Separation
The dashboard fetches scenarios dynamically from the API and computes live metrics entirely client-side:
- **Live ROC-AUC:** A continuous, tie-aware Rank-Sum calculation.
- **Score-Separation Histogram:** A bespoke back-to-back bar chart designed to visualize the exact prediction distributions and recreate the same class-separation signature previously found via manual eyeball testing (e.g. `0.8390` vs `0.2073`), perfectly illustrating the ranking success and absolute-scale compression of the GNN.

---

## 5. Dataset Generations & Evaluation

### 5.1 Phase 1 Baseline (random origins, pre-Phase-2-fixes)

Initial architecture evaluation on a purely random-origin synthetic network with distance heuristics (prior to traffic integration and topology serialization bug fixes):

| Model | ROC-AUC |
| :--- | :--- |
| **GNN (CascadeNet)** | **0.9004** |
| Distance Heuristic | 0.8910 |
| MLP (No Message Passing) | 0.9099 |

> [!NOTE]
> The dataset in Phase 1 utilized unlimited randomly-sampled origins and had a median cascade size of 8.44%. The architecture suffered from a data leakage bug via `gtfs_parser.py` (weight pollution) and failed to serialize node ordering properly, which have since been resolved.

### 5.2 Phase 2 ICS-Driven Results (post traffic/topology fixes)

Following the remediation of the traffic-parser weight pollution and topology serialization bugs, the simulation engine was integrated with specific Industrial Control System (ICS) telemetry points. 

**N=3 Baseline Performance (Mean ± Std over Seeds 0, 1, 2):**

| Model | ROC-AUC | Unbatched Precision (P@K) |
| :--- | :--- | :--- |
| **GNN (CascadeNet)** | **0.9089 ± 0.0003** | **0.6790 ± 0.0060** |
| Distance Heuristic | 0.8991 | 0.6601 |
| MLP (with Distance) | 0.9014 | 0.6592 |

> [!WARNING]
> **Negative Constraints on Dataset Realism**
> 1. **Synthetic Cross-Layer Edges:** Connections bridging the physical power grid to the transit layer are completely synthetic. There is no real-world topology mapping substations to specific traffic lights; the connectivity was procedurally generated using spatial distance thresholds.
> 2. **Traffic Layer is Feature-Only:** The `gtfs_parser.py` integration provides static, isolated node features for the transit network. The physics engine **does not actually simulate propagation** through the traffic layer — it only propagates through the power lines. The GNN relies heavily on the shortest-path distance `Feature 8`, meaning the traffic layer adds essentially no dynamic value to the cascade predictions.
> 3. **ICS/SCADA Diversity Limitation:** Although driven by real telemetry, the entire training corpus maps back to **only 4 unique ICS origins** (`PWR_111`, `PWR_112`, `PWR_113`, `PWR_114`). The model is highly overfit to the cascade footprints of these 4 origins. It does not generalize to novel attacks originating from unseen parts of the grid.

> [!TIP]
> **Expected Near-Zero Variance**: The extremely tight AUC variance (±0.0003) across N=3 independent training seeds is a direct consequence of the 4-origin limitation. Because the model only ever sees cascades originating from one of four static grid points, the learning task is highly constrained. Different random initialization seeds converge to nearly identical predictive functions for these 4 footprints, resulting in artificially low variance that should not be interpreted as robust generalization.

> [!NOTE]
> All results in Section 5.1 and 5.2 above utilized the 1000m distance threshold for building the fused graph (i.e. power/water nodes were linked to the nearest road node within 1000m).

---

## 6. Phase 2: Resolving Core Proposal Claims

Following the core architectural build, we rigorously closed out the remaining claims outlined in the original research proposal.

### 6.1 Formal Architectural Ablation
We formally evaluated the efficacy of fusing the GNN with the classical SEIR simulator (on a 50-graph test subset at 1000m threshold). Literal output from `evaluation/ablation.py`:

```
Model                | ROC-AUC    | MAE (Time)
------------------------------------------------------------
GNN Only             | 0.6941     | 164.27327
SEIR Only            | 0.8977     | N/A (non-temporal)
Hybrid Engine        | 0.5393     | 190.14215
```

**Insight**: The Hybrid Engine's severe collapse to 0.53 AUC highlights a crucial architectural mismatch. The GNN's `edge_probs` are trained exclusively when the `--train-hybrid` objective is active. Because the provided `best_model.pth` checkpoint was trained purely on node-level classification/ranking, the edge transmission probabilities emitted by the GNN are effectively untrained noise. When this noise is fed into a strict Monte Carlo SEIR simulation as literal Markov transition probabilities, the predicted cascade diverges completely from ground truth.

### 6.2 Chokepoint Intervention Ranking
We successfully implemented a `GreedyChokepointRanker` which uses the trained GNN to identify optimal nodes for immunization. Rather than evaluating every possible permutation, it leverages a fast, degree-pruned forward pass to evaluate the marginal risk reduction of immunizing specific candidate nodes. This feature is exposed via a dedicated `/api/chokepoint/rank` endpoint (`backend/routers/chokepoint.py`) and integrated directly into the React dashboard (`frontend/src/App.jsx`, verified by code search).

### 6.3 Digital-Twin Validation Loop
We built a validation harness (`evaluation/digital_twin_validation.py`) to test whether GNN-predicted failure nodes cause measurable structural degradation when applied to the city graph. Because Mininet is not installed in the Windows environment, the script runs in **dry-run mode** — it logically removes GNN-predicted failed nodes from the NetworkX city graph and measures connectivity loss.

Literal output from a direct run:
```
[Base State] Graph is partitioned into 4 components. Largest component size: 2275
[Degraded State] After removing predicted failed nodes:
Graph is partitioned into 4 components. Largest component size: 2272

[Result] GNN-predicted cascade results in a 0.13% loss of global reachability.
```

**Honest Assessment**: At the 0.5 probability threshold, the trained GNN predicted only 3 nodes as failed in the test scenario. Removing 3 non-hub nodes from a 2,278-node graph causes negligible structural disruption (0.13%). Full Mininet SDN emulation in a Linux/WSL2 environment with a model trained on the hybrid edge objective would be required to produce a meaningful structural disruption result.

---

## Conclusion

The Cascade Shield project represents a rigorously self-audited implementation of a GNN-driven cascade failure prediction pipeline on a 2,278-node tri-layer infrastructure graph. We distinguish between two tracked runs:

1. **Phase 1 Baseline:** The original, buggy dataset utilizing randomly sampled origins, generating **0.9004 ± 0.0010 ROC-AUC**.
2. **Phase 2 ICS-Driven:** The finalized, bug-free dataset utilizing explicit real-world ICS telemetry target nodes (`PWR_111`-`114`), generating **0.9089 ± 0.0003 ROC-AUC** and **0.6790 ± 0.0060 P@K**.

However, three structural facts dictate the interpretation of this final headline result:
1. **The Distance Dominance:** A naive Dijkstra shortest-path distance heuristic achieves 0.8991 AUC and 0.6601 P@K on the identical Phase 2 dataset. The GNN's 0.9089 AUC is a marginal (+0.0098) improvement over hardcoded topological proximity. Similarly, a feedforward MLP given the distance feature achieved 0.9014 AUC.
2. **Message Passing as Structural Recovery:** When the topological distance feature is explicitly removed from the dataset (Feature 8 Ablation), a fully retrained GNN achieves **0.8425 and 0.8314 AUC** across 2 independent seeds (with P@K of 0.6366 and 0.6035). While this falls behind the naive heuristic, it is drastically higher than the ~0.56 AUC achieved by a distance-free MLP baseline. This reveals that the GNN's message passing is not entirely redundant—when explicit distance is withheld, the graph convolution powerfully recovers and approximates that structural proximity on its own. However, this recovery introduces two distinct forms of instability:
    * **Within-run non-convergence:** Unlike the baseline, the ablated models do not smoothly converge within 100 epochs, oscillating violently around their peak validation scores right up to the epoch limit.
    * **Seed-to-seed variance:** The lack of convergence drives the Δ between seeds to >0.011 (roughly 40x higher than the baseline's ±0.0003 variance). 
    * *(Limitation: Results are reported at a 100-epoch budget; whether the ablated architecture converges to a higher, more stable ceiling given 150-200 epochs was not tested).*
3. **Feature-Only Traffic:** The traffic network ingest serves only as static feature padding. The SEIR simulator propagates exclusively along power edges, meaning cross-layer dependencies add no dynamic predictive value.
4. **The 4-Origin Artifact:** Because the ICS telemetry traces back to only 4 unique nodes out of 2,278, the dataset is hyper-constrained. The 0.0003 standard deviation across different N=3 baseline seeds is not robust generalization, but rather the mathematical certainty of different weight initializations learning the exact same 4 spatial footprints.

The honest scientific conclusion is more nuanced than initially hypothesized: **Distance is the single strongest predictor of cascade propagation on this planar graph. However, when distance is explicitly withheld, message passing successfully substitutes for it and recovers substantial signal (0.56 → ~0.83/0.84 AUC) that a feedforward network cannot.** The engineering contributions — the tri-layer graph fusion pipeline, the simulation infrastructure, the SCADA dashboard, and the rigorous audit methodology — remain substantial. The scientific framing should reflect that message passing is highly effective at structural recovery, even if handing it distance directly yields a higher, more stable absolute bound.


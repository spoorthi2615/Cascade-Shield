# Cascade Simulator Assumptions & Parameter Rationale

## Propagation Probabilities
The base propagation probabilities defined in `propagation_rules.py` determine the likelihood that an `INFECTED` node will compromise a `SUSCEPTIBLE` neighbor along a specific edge type during a single tick.

**Assigned Values:**
- `PHYSICAL`: 0.65 (Latency: 2 ticks)
- `LOGICAL`: 0.80 (Latency: 1 tick)
- `INFORMATIONAL`: 0.95 (Latency: 1 tick)

**Rationale:**
These probabilities were calibrated to target a moderate-severity cascade reach fraction (e.g., ~10-15% of total nodes) on a realistically sized, heterogeneous topology. This baseline is designed to avoid degenerate datasets where attacks either never propagate beyond the origin (0 hops) or always cause 100% total systemic collapse. 

In calibration runs on the IEEE 123-bus test feeder (N=132), applying these baseline probabilities inside a continuous-retry SEIR model yielded an average cascade size of ~12.4 nodes (~9.3% reach), with a diverse long-tail distribution generating cascades ranging from 1 to 47 nodes. This ensures CascadeNet has a balanced, varied training dataset to learn chokepoint rankings from.

## State Transitions & Recovery
- **Continuous Propagation:** An `INFECTED` node acts as a continuous threat, attempting to infect all its `SUSCEPTIBLE` neighbors at every discrete tick it remains in the `INFECTED` state. 
- **Recovery Threshold:** By default, nodes with a `criticality >= 0.5` transition to a permanent `FAILED` state after infection, simulating critical infrastructure damage. Nodes with `criticality < 0.5` transition to `RECOVERED` after a fixed window, representing successful incident response and containment on non-critical components.

## Synthetic Subgraphs & Cross-Subsystem Edges
To generate a comprehensive cyber-physical dataset, the base OpenDSS power feeder graphs are extended with synthetic Water, Traffic, and Transit subgraphs. 
- **Water Network:** Generated via random geometric graph, sized proportionally (~30%) to the power graph.
- **Traffic Network:** Generated via 2D grid graph, sized proportionally (~40%) to the power graph.
- **Transit Network:** Generated via Watts-Strogatz small-world graph, sized proportionally (~20%) to the power graph.

**Cross-Subsystem Edge Placement Methodology:**
Rather than random arbitrary connectivity, cross-subsystem edges are placed deterministically based on logical real-world dependencies to maintain dataset integrity:
1. **Power-to-Water (PHYSICAL):** Every highest-degree node (pump station hub) in the water network is strictly connected to the nearest topological power substation/bus via a `PHYSICAL_DEPENDENCY` edge.
2. **Power-to-Traffic (PHYSICAL):** Traffic controllers are clustered topologically. Each cluster centroid receives a `PHYSICAL_DEPENDENCY` edge from the power graph to simulate grid-tied intersections.
3. **Transit-to-Power (LOGICAL/INFORMATIONAL):** Transit hubs are connected to power substations via `LOGICAL` (shared SCADA backhaul) and `INFORMATIONAL` (shared credentials) edges, representing centralized city operations centers.

The total number of cross-subsystem edges is parameterized by `density_param`, but the transmission probability across them is modulated by an independent `cross_weight` coupling factor.

During calibration on the 246-node city graph, we found a **structural floor** for topology: placing exactly 1 edge per category (3 total cross-domain edges, `density_param=0.01`) represents the absolute minimum connectivity required to link a 4-subsystem city. However, if those 3 edges transmit at full base probability (`cross_weight=1.0`), exactly 42.6% of multi-node cascades cross domains—far exceeding our 15-25% target due to the high saturation of the small non-power subgraphs.

**Calibration Decision:** To achieve our target 15-25% cross-domain fraction without disconnecting subsystems, we maintain the structural floor of 3 edges but dial down their coupling strength. By setting `cross_weight=0.30` on these specific edges (simulating restricted logical access or tighter physical security perimeters at cross-domain boundaries), the cross-domain cascade fraction drops to exactly **18.1%**, landing perfectly in our target band.


## Feature Engineering: Distance-to-Origin
- **Relative vs Absolute**: We compute the shortest-path distance from the infection origin using propagation-weighted edge costs (1.0 / (edge_weight + 1e-5)). To stabilize the GNN inputs across diverse infrastructure footprints (dense vs sparse), these raw distances are Min-Max normalized to [0, 1] on a *per-graph* basis.
- **Interpretation Consequence**: Because normalization happens per-graph, the distance feature encodes *relative position* within that specific scenario's topology, rather than an absolute comparable distance across scenarios. We explicitly chose this relative encoding so the model learns structural ranking rather than memorizing absolute path bounds.
- **Unreachable Nodes**: Nodes completely disconnected from the origin in a specific scenario are assigned a pre-normalization distance equal to the maximum finite distance in that graph. This maps them to 1.0 (maximally far) after normalization.

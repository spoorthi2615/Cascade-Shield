# Cascade Shield

**Cascade Shield** is an infrastructure cascade failure prediction system. It models how catastrophic failures (e.g., cyberattacks, physical faults) propagate across highly interconnected, multi-subsystem city grids (Power, Water, Traffic, Transit). 

By benchmarking Graph Neural Networks (GNNs) against Classical SEIR epidemiological models and zero-parameter topological heuristics, this project rigorously investigates the representational limits of graph learning on deterministic anomaly propagation.

## 🚀 Key Features

- **Hybrid Topological Generation:** Fuses real, standardized IEEE Power Test Feeders (via OpenDSS) with synthetic SCADA networks and IoT cross-dependencies.
- **SEIR Discrete Event Simulator:** Models continuous-time failure propagation across the resulting heterogeneous graph.
- **Pairwise Ranking Loss:** A bespoke GNN training paradigm designed to combat extreme class imbalance (91.8% safe nodes) by learning relative topological risk rather than absolute probabilities.
- **Interactive SCADA Dashboard:** A React-based, control-room styled visualizer built with `react-force-graph-2d` for high-performance canvas rendering of complex cascades.

## 📊 Evaluation & Results (Phase 2)

Following a rigorous Phase 2 audit, the pipeline evaluates cascade propagation on a 2,278-node tri-layer graph using real-world ICS telemetry origins (`PWR_111`-`114`).

- **GNN Representational Ceiling:** `0.9089 ± 0.0003` ROC-AUC | `0.6790 ± 0.0060` P@K
- **Feedforward MLP (w/ Distance):** `0.9014` ROC-AUC
- **Naive Distance Heuristic:** `0.8991` ROC-AUC | `0.6601` P@K
- **GNN Feature 8 Ablation (No Distance):** `0.83-0.84` ROC-AUC (Unstable across seeds)
- **MLP Ablation (No Distance):** `~0.56` ROC-AUC

**Key Scientific Findings:**
1. **Distance Dominance:** Topological shortest-path distance is the single strongest predictor of cascade propagation, achieving 0.8991 AUC purely via heuristic. The GNN's 0.9089 AUC is a marginal (+0.0098) improvement over this hardcoded topological proximity.
2. **Message Passing as Structural Recovery:** When explicit distance is withheld, a standard MLP collapses to ~0.56 AUC. The GNN, however, powerfully recovers and approximates that structural proximity via message passing, lifting the AUC back to ~0.83-0.84, albeit exhibiting high instability and failing to smoothly converge within standard epoch budgets.

## 📁 Repository Structure

```
├── backend/            # FastAPI inference server serving scenarios and predictions
├── baselines/          # Classical SEIR and distance-heuristic implementations
├── data/               # Raw IEEE OpenDSS feeder files
├── docs/               # Architecture documents and related work
├── frontend/           # React + Vite interactive SCADA dashboard
├── graph/              # Topology synthesis from OpenDSS to NetworkX/Neo4j
├── models/             # PyTorch Geometric GNN architecture and training scripts
├── scripts/            # Notebooks and evaluation runners
└── simulator/          # SimPy-based Discrete Event Simulator for cascading failures
```

## 🛠️ Quickstart

### 1. Backend (FastAPI)
The backend requires Python 3.9+ and PyTorch.
```bash
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```
*The API will be available at `http://localhost:8000`.*

### 2. Frontend (React / Vite)
The frontend uses standard Node/npm tooling.
```bash
cd frontend
npm install
npm run dev
```
*The dashboard will hot-reload at `http://localhost:5173`.*

## 🔬 Reproducibility

The repository is built for strict reproducibility. See `REPRODUCIBILITY.md` for exact commands to recreate the 100-epoch training sweeps, capacity ablation tests, and seed controls used to establish the representation ceiling.

---
*Built as a state-of-the-art framework for future graph-based infrastructure failure modeling.*

# CascadeShield

**CascadeShield** is an infrastructure cascade failure prediction system. It models how catastrophic failures (e.g., cyberattacks, physical faults) propagate across highly interconnected, multi-subsystem city grids (Power, Water, Traffic, Transit). 

By benchmarking Graph Neural Networks (GNNs) against Classical SEIR epidemiological models and zero-parameter topological heuristics, this project rigorously investigates the representational limits of graph learning on deterministic anomaly propagation.

## 🚀 Key Features

- **Hybrid Topological Generation:** Fuses real, standardized IEEE Power Test Feeders (via OpenDSS) with synthetic SCADA networks and IoT cross-dependencies.
- **SEIR Discrete Event Simulator:** Models continuous-time failure propagation across the resulting heterogeneous graph.
- **Pairwise Ranking Loss:** A bespoke GNN training paradigm designed to combat extreme class imbalance (91.8% safe nodes) by learning relative topological risk rather than absolute probabilities.
- **Interactive SCADA Dashboard:** A React-based, control-room styled visualizer built with `react-force-graph-2d` for high-performance canvas rendering of complex cascades.

## 📊 Evaluation & Results

Extensive robustness testing (learning rate sweeps, capacity scaling up to 128-dim, and multi-seed controls) revealed a strict representational ceiling:
- **GNN Representational Ceiling:** `0.735 ± 0.033` ROC-AUC.
- **Classical SEIR Model:** `0.9576` ROC-AUC.
- **Naive Distance Heuristic:** `0.9480` ROC-AUC.

The GNN, while outperforming static centrality measures, hits a hard ceiling bounded by topological shortest-paths out-of-sample.

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

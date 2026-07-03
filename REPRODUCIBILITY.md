# Reproducibility Guide: Representational Ceiling Results

This document contains the exact commands used to generate the final performance ceiling numbers for Cascade Shield.

## 1. Naive Distance Heuristic
The structural zero-parameter heuristic (inversely proportional to propagation-weighted shortest path) is hardcoded directly into the evaluation script. It acts as the performance upper bound (`0.9480` ROC-AUC) given the simulator's deterministic retry mechanics.
To reproduce this evaluation:
```bash
python scripts/run_evaluations.py
```

## 2. GNN Ceiling (6-Seed Capacity Control)
To rigorously estimate the GNN's predictive ceiling and eliminate "winner's curse" bias, we ran 3 explicit seeds (`0, 1, 2`) at both the baseline capacity (`hidden_dim=64`) and a doubled capacity (`hidden_dim=128`).
We utilized a Pairwise Ranking Loss (`lambda-rank=0.05`) and a reduced, stable learning rate (`lr=0.0001`).

### Training Commands
You can reproduce the 6 training runs by executing the following commands sequentially (or running the provided `run_seeds.ps1` script):

**Dim 64:**
```bash
python -u models/train.py --epochs 100 --num-layers 3 --lambda-rank 0.05 --lr 0.0001 --hidden-dim 64 --seed 0
python -u models/train.py --epochs 100 --num-layers 3 --lambda-rank 0.05 --lr 0.0001 --hidden-dim 64 --seed 1
python -u models/train.py --epochs 100 --num-layers 3 --lambda-rank 0.05 --lr 0.0001 --hidden-dim 64 --seed 2
```

**Dim 128:**
```bash
python -u models/train.py --epochs 100 --num-layers 3 --lambda-rank 0.05 --lr 0.0001 --hidden-dim 128 --seed 0
python -u models/train.py --epochs 100 --num-layers 3 --lambda-rank 0.05 --lr 0.0001 --hidden-dim 128 --seed 1
python -u models/train.py --epochs 100 --num-layers 3 --lambda-rank 0.05 --lr 0.0001 --hidden-dim 128 --seed 2
```
*Note: Checkpoints are durably saved to `checkpoints/width_{dim}_seed_{seed}/best_model.pth` to prevent overwriting.*

### Evaluation Command
To evaluate the 6 trained checkpoints on the test set and calculate the mean ± std AUC and Val-Test Gap, run the dedicated evaluation script:
```bash
python evaluate_seeds.py
python check_gap.py
```

### Final Results
**Dim 64:** `0.7420 ± 0.0212` Test AUC
**Dim 128:** `0.7280 ± 0.0414` Test AUC
**Mean Val-Test Gap (Winner's Curse):** `0.0498 ± 0.0338` (All 6 runs exhibit a systematic positive gap between validation AUC and out-of-sample test AUC).

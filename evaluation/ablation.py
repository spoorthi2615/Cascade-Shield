import os
import sys
import torch
import csv
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cascade_net import CascadeNet
from models.evaluate import evaluate
from models.train import load_datasets
from baselines.classical_seir import ClassicalSEIRPredictor
from models.hybrid_engine import HybridSEIRPredictor
from scripts.run_evaluations import generate_static_topology

def run_ablation():
    print("============================================================")
    print(" CASCADE SHIELD ABLATION STUDY ")
    print("============================================================")
    
    # 1. Setup Data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "1000m")
    city_G, node_to_idx, idx_to_node = generate_static_topology(1000)
    _, _, test_graphs = load_datasets(data_dir, smoke_test=False)
    
    # Run on a smaller subset (e.g. 50 graphs) to make it reasonably fast
    num_eval = min(50, len(test_graphs))
    test_loader = DataLoader(test_graphs[:num_eval], batch_size=1, shuffle=False)
    print(f"Evaluating on {num_eval} test graphs from {data_dir}...\n")
    
    in_channels = test_graphs[0].x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Setup GNN (CascadeNet)
    hidden_dim = 64
    gnn_model = CascadeNet(in_channels=in_channels, edge_dim=4, hidden_dim=hidden_dim, num_layers=3, heads=4, use_supernode=True)
    best_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "width_64_seed_None", "best_model.pth")
    
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
        gnn_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded CascadeNet from {best_path}")
    else:
        print("WARNING: No trained CascadeNet checkpoint found. Using untrained architecture.")
    
    gnn_model.to(device)

    # 3. Setup SEIR Only (Classical SEIR)
    seir_only = ClassicalSEIRPredictor(num_trials=10) # 10 trials for speed
    seir_only.G = city_G
    seir_only.node_to_idx = node_to_idx
    seir_only.idx_to_node = idx_to_node

    # 4. Setup Hybrid Engine (GNN + SEIR)
    hybrid_model = HybridSEIRPredictor(model=gnn_model, num_trials=10)
    hybrid_model.G = city_G
    hybrid_model.node_to_idx = node_to_idx
    hybrid_model.idx_to_node = idx_to_node

    # 5. Evaluate
    results = {}

    print("\n[1/3] Evaluating GNN Only (CascadeNet Direct Inference)...")
    res_gnn = evaluate(gnn_model, test_loader, device=device, is_baseline=False)
    results['GNN Only'] = res_gnn

    print("\n[2/3] Evaluating SEIR Only (Classical Physics)...")
    res_seir = evaluate(seir_only, test_loader, device=device, is_baseline=True)
    results['SEIR Only'] = res_seir

    print("\n[3/3] Evaluating Hybrid Engine (GNN-driven SEIR)...")
    res_hybrid = evaluate(hybrid_model, test_loader, device=device, is_baseline=True)
    results['Hybrid Engine'] = res_hybrid

    # 6. Print Summary Table
    print("\n============================================================")
    print(f"{'Model':<20} | {'ROC-AUC':<10} | {'MAE (Time)':<10}")
    print("------------------------------------------------------------")
    for name, res in results.items():
        auc = f"{res['ROCAUC']:.4f}"
        mae = res['Time_MAE']
        mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
        print(f"{name:<20} | {auc:<10} | {mae_str:<10}")
    print("============================================================\n")

if __name__ == "__main__":
    run_ablation()

import os
import sys
import torch
import csv
import numpy as np
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cascade_net import CascadeNet
from models.evaluate import evaluate
from models.train import load_datasets
from graph.build_topology import parse_opendss_feeder, build_networkx_graph, generate_synthetic_city
from baselines.classical_seir import ClassicalSEIRPredictor
from baselines.static_centrality import StaticCentralityPredictor
from baselines.isolated_anomaly_detector import IsolatedAnomalyPredictor

def generate_static_topology():
    """Rebuilds the exact topology used during dataset generation."""
    feeder_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "123Bus", "IEEE123Master.dss")
    nodes, edges = parse_opendss_feeder(feeder_path)
    base_power_G = build_networkx_graph(nodes, edges)
    for n in base_power_G.nodes():
        base_power_G.nodes[n]["subsystem"] = "power"
    city_G = generate_synthetic_city(base_power_G, density_param=0.01, seed=42)
    
    # We must match the node ordering exactly as serialized in create_base_tensors
    node_to_idx = {n: i for i, n in enumerate(city_G.nodes())}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    
    return city_G, node_to_idx, idx_to_node

def run_all_evaluations(args):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    _, _, test_graphs = load_datasets(data_dir, smoke_test=False)
    
    if args.smoke_test:
        test_graphs = test_graphs[:30] # 30-graph smoke test to get K=0 samples
        print(f"--- SMOKE TEST MODE: Evaluating on {len(test_graphs)} test graphs ---")
    else:
        print(f"--- Loaded {len(test_graphs)} test graphs ---")
        
    in_channels = test_graphs[0].x.shape[1]
    assert in_channels == 9, f"Expected 9-dim features (origin-conditioned + distance), but got {in_channels}"
    
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    G, node_to_idx, idx_to_node = generate_static_topology()
    
    results = {}
    
    # 1. CascadeNet (Loaded from Checkpoint)
    device = torch.device('cpu')
    hidden_dim = getattr(args, 'hidden_dim', 64)
    model = CascadeNet(in_channels=in_channels, edge_dim=4, hidden_dim=hidden_dim, num_layers=args.num_layers, heads=4, use_supernode=args.use_supernode)
    best_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pth")
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "checkpoint_latest.pth")
    
    loaded_path = None
    if os.path.exists(best_path):
        loaded_path = best_path
    elif os.path.exists(ckpt_path):
        loaded_path = ckpt_path
        
    if loaded_path:
        ckpt = torch.load(loaded_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"CascadeNet loaded from checkpoint {os.path.basename(loaded_path)} (Epoch {ckpt['epoch']}).")
    else:
        print("WARNING: No CascadeNet checkpoint found. Evaluating untrained architecture.")
    
    print("\nEvaluating CascadeNet (Untrained)...")
    model_untrained = CascadeNet(in_channels=in_channels, edge_dim=4, hidden_dim=hidden_dim, num_layers=args.num_layers, heads=4, use_supernode=args.use_supernode)
    model_untrained.to(device)
    res_untrained = evaluate(model_untrained, test_loader, device=device, is_baseline=False)
    results['CascadeNet (Untrained)'] = res_untrained
    print(res_untrained)

    print("\nEvaluating CascadeNet (Trained)...")
    res_cas = evaluate(model, test_loader, device=device, is_baseline=False)
    sample_g = test_graphs[0]
    with torch.no_grad():
        log_c, _ = model(sample_g.x, sample_g.edge_index, sample_g.edge_attr, getattr(sample_g, 'batch', None))
        p = torch.sigmoid(log_c).cpu().numpy().flatten()
        y = sample_g.y.cpu().numpy()
        print("--- CascadeNet Prediction Eyeball Test (Graph 0) ---")
        print(f"Top 5 predicted node scores: {sorted(p, reverse=True)[:5]}")
        print(f"True cascade size: {y.sum()}")
        infected_scores = p[y == 1]
        safe_scores = p[y == 0]
        print(f"Mean score on Infected nodes: {infected_scores.mean() if len(infected_scores) > 0 else 0:.4f}")
        print(f"Mean score on Safe nodes:     {safe_scores.mean() if len(safe_scores) > 0 else 0:.4f}")
        
    results['CascadeNet'] = res_cas
    print(res_cas)
    
    # 2. Baseline Predictors
    def init_baseline(baseline_class, **kwargs):
        b = baseline_class(**kwargs)
        b.G = G
        b.node_to_idx = node_to_idx
        b.idx_to_node = idx_to_node
        return b
        
    seir = init_baseline(ClassicalSEIRPredictor, num_trials=20)
    seir.is_temporal = True
    print("\nEvaluating Classical SEIR...")
    res_seir = evaluate(seir, test_loader, is_baseline=True)
    results['Classical SEIR'] = res_seir
    print(res_seir)
    
    from baselines.classical_seir import BlindClassicalSEIRPredictor
    blind_seir = init_baseline(BlindClassicalSEIRPredictor, num_trials=10)
    print("\nEvaluating Blind Classical SEIR (Mixture Prior)...")
    res_blind_seir = evaluate(blind_seir, test_loader, is_baseline=True)
    results['Blind SEIR'] = res_blind_seir
    print(res_blind_seir)
    
    # GNN-SEIR Hybrid Evaluation (Step 2)
    hybrid_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model_hybrid.pth")
    hybrid_seir = None
    if os.path.exists(hybrid_path):
        print("\nEvaluating GNN-SEIR Hybrid (Origin-Blind)...")
        # Load hybrid model checkpoint (always 3 layers, no supernode, edge head enabled)
        model_hybrid = CascadeNet(in_channels=in_channels, edge_dim=4, hidden_dim=64, num_layers=3, heads=4, use_supernode=False)
        hybrid_ckpt = torch.load(hybrid_path, map_location=device, weights_only=True)
        try:
            model_hybrid.load_state_dict(hybrid_ckpt['model_state_dict'])
            model_hybrid.eval()
            
            from models.hybrid_engine import HybridSEIRPredictor
            hybrid_seir = init_baseline(HybridSEIRPredictor, model=model_hybrid, num_trials=10)
            res_hybrid = evaluate(hybrid_seir, test_loader, is_baseline=True)
            results['GNN-SEIR Hybrid'] = res_hybrid
            print(res_hybrid)
            
            # Log diagnostic metrics
            if len(hybrid_seir.diag_origin_infected) > 0:
                origin_inf_acc = np.mean(hybrid_seir.diag_origin_infected) * 100
                origin_match_acc = np.mean(hybrid_seir.diag_origin_matched) * 100
                print(f"--- GNN-SEIR Hybrid Diagnostics ---")
                print(f"  Origin Pick is Infected:  {origin_inf_acc:.2f}%")
                print(f"  Origin Pick matches True: {origin_match_acc:.2f}%")
        except RuntimeError as e:
            print("Skipping Hybrid evaluation due to dimension mismatch (Phase 1 7-dim checkpoint loaded in Phase 2 8-dim schema).")

    class NaiveDistancePredictor:
        def __init__(self):
            self.is_temporal = False
            
        def fit(self, train_loader):
            pass
            
        def predict(self, graph, origin_node=None):
            import networkx as nx
            
            # Compute shortest paths using the same cost function as train.py
            # 1.0 / (weight + 1e-5)
            # In NetworkX, we need a weight function
            def weight_func(u, v, d):
                # Cascade Shield sets weight in d['weight']
                w = d.get('weight', 1.0)
                return 1.0 / (w + 1e-5)
                
            dist_dict = nx.single_source_dijkstra_path_length(graph, origin_node, weight=weight_func)
            
            # Find max finite distance
            max_finite = max(dist_dict.values()) if dist_dict else 0.0
            
            risk_dict = {}
            for node in graph.nodes():
                if node in dist_dict:
                    dist = dist_dict[node]
                else:
                    dist = max_finite
                
                # Naive risk is 1.0 / (dist + 1.0)
                risk_dict[node] = 1.0 / (dist + 1.0)
                
            return risk_dict

    naive_dist = init_baseline(NaiveDistancePredictor)
    print("\nEvaluating Naive Distance Heuristic...")
    res_naive_dist = evaluate(naive_dist, test_loader, is_baseline=True)
    results['Naive Distance'] = res_naive_dist
    print(res_naive_dist)
            
    centrality = init_baseline(StaticCentralityPredictor)
    centrality.is_temporal = False
    print("\nEvaluating Static Centrality...")
    res_cent = evaluate(centrality, test_loader, is_baseline=True)
    results['Static Centrality'] = res_cent
    print(res_cent)
    
    anomaly = init_baseline(IsolatedAnomalyPredictor)
    anomaly.is_temporal = False
    print("\nEvaluating Isolated Anomaly...")
    res_anom = evaluate(anomaly, test_loader, is_baseline=True)
    results['Isolated Anomaly'] = res_anom
    print(res_anom)
    
    # Paired Bootstrap Significance Testing against Blind Classical SEIR
    if 'Blind SEIR' in results:
        print("\n=== Paired Bootstrap Significance Test (GNN vs. Blind SEIR) ===")
        
        # Precompute predictions on all test graphs to speed up bootstrapping
        print("Precomputing predictions...")
        true_labels_list = []
        pred_base_list = []  # Will store Blind SEIR predictions
        pred_new_list = []   # Will store CascadeNet predictions
        
        for g in test_graphs:
            g = g.to(device)
            # Call Blind SEIR predict
            # We map origin node name just to satisfy the predict contract (ignored internally by Blind SEIR)
            valid_times = g.y_time.clone()
            valid_times[g.y_time < 0] = float('inf')
            origin_idx = torch.argmin(valid_times).item()
            origin_name = blind_seir.idx_to_node[origin_idx]
            
            risk_dict = blind_seir.predict(blind_seir.G, origin_name)
            pred_base = torch.zeros(g.num_nodes, dtype=torch.float)
            for name, score in risk_dict.items():
                idx = blind_seir.node_to_idx[name]
                pred_base[idx] = score
                
            with torch.no_grad():
                logits_new, _ = model(g.x, g.edge_index, g.edge_attr, getattr(g, 'batch', None))
                
            true_labels_list.append(g.y.cpu().numpy())
            pred_base_list.append(pred_base.cpu().numpy().flatten())
            pred_new_list.append(torch.sigmoid(logits_new).cpu().numpy().flatten())
                
        # Bootstrap loop with a fixed seed
        print("Running paired bootstrap (1000 iterations)...")
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(42)
        M = len(test_graphs)
        
        delta_aucs = []
        delta_pks = []
        
        for _ in range(1000):
            # Sample scenario indices with replacement
            indices = rng.choice(M, size=M, replace=True)
            
            # Concatenate for global AUC
            y_true_boot = np.concatenate([true_labels_list[i] for i in indices])
            y_pred_base_boot = np.concatenate([pred_base_list[i] for i in indices])
            y_pred_new_boot = np.concatenate([pred_new_list[i] for i in indices])
            
            auc_base = roc_auc_score(y_true_boot, y_pred_base_boot)
            auc_new = roc_auc_score(y_true_boot, y_pred_new_boot)
            delta_aucs.append(auc_new - auc_base)
            
            # Average Precision@K over K > 1 scenarios in the boot sample
            pk_base_scores = []
            pk_new_scores = []
            for i in indices:
                y = true_labels_list[i]
                K = int(y.sum())
                if K > 1:
                    top_k_base = np.argsort(pred_base_list[i])[-K:]
                    pk_base_scores.append(np.sum(y[top_k_base] == 1.0) / K)
                    
                    top_k_new = np.argsort(pred_new_list[i])[-K:]
                    pk_new_scores.append(np.sum(y[top_k_new] == 1.0) / K)
                    
            delta_pks.append(np.mean(pk_new_scores) - np.mean(pk_base_scores))
            
        print(f"Point Estimate baseline AUC: {res_blind_seir.get('ROCAUC', 0.0):.4f}")
        print(f"Point Estimate new AUC:      {res_cas.get('ROCAUC', 0.0):.4f}")
        
        ci_auc = (np.percentile(delta_aucs, 2.5), np.percentile(delta_aucs, 97.5))
        ci_pk = (np.percentile(delta_pks, 2.5), np.percentile(delta_pks, 97.5))
        
        print(f"\nDelta ROC-AUC: Mean={np.mean(delta_aucs):.4f}, 95% CI=({ci_auc[0]:.4f}, {ci_auc[1]:.4f})")
        print(f"Delta Precision@K: Mean={np.mean(delta_pks):.4f}, 95% CI=({ci_pk[0]:.4f}, {ci_pk[1]:.4f})")
        
        # Win conditions
        win_auc = ci_auc[0] > 0 and np.mean(delta_aucs) >= 0.015
        win_pk = ci_pk[0] > 0 and np.mean(delta_pks) >= 0.02
        
        print(f"ROC-AUC Significance Win: {win_auc} (Need: Mean >= +0.015 and lower CI bound > 0)")
        print(f"Precision@K Significance Win: {win_pk} (Need: Mean >= +0.02 and lower CI bound > 0)")

    # Paired Bootstrap: GNN-SEIR Hybrid vs GNN Baseline
    if hybrid_seir is not None and 'GNN-SEIR Hybrid' in results:
        print("\n=== Paired Bootstrap Significance Test (Hybrid vs. GNN Baseline) ===")
        print("Precomputing predictions...")
        true_labels_list = []
        pred_base_list = []  # Will store GNN Baseline predictions
        pred_new_list = []   # Will store Hybrid predictions
        
        for g in test_graphs:
            g = g.to(device)
            with torch.no_grad():
                logits_base, _ = model(g.x, g.edge_index, g.edge_attr, getattr(g, 'batch', None))
                
            valid_times = g.y_time.clone()
            valid_times[g.y_time < 0] = float('inf')
            origin_idx = torch.argmin(valid_times).item()
            origin_name = hybrid_seir.idx_to_node[origin_idx]
            
            risk_dict = hybrid_seir.predict(hybrid_seir.G, origin_name)
            pred_hybrid = torch.zeros(g.num_nodes, dtype=torch.float)
            for name, score in risk_dict.items():
                idx = hybrid_seir.node_to_idx[name]
                pred_hybrid[idx] = score
                
            true_labels_list.append(g.y.cpu().numpy())
            pred_base_list.append(torch.sigmoid(logits_base).cpu().numpy().flatten())
            pred_new_list.append(pred_hybrid.cpu().numpy().flatten())
            
        print("Running paired bootstrap (1000 iterations)...")
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(42)
        M = len(test_graphs)
        
        delta_aucs = []
        delta_pks = []
        
        for _ in range(1000):
            indices = rng.choice(M, size=M, replace=True)
            y_true_boot = np.concatenate([true_labels_list[i] for i in indices])
            y_pred_base_boot = np.concatenate([pred_base_list[i] for i in indices])
            y_pred_new_boot = np.concatenate([pred_new_list[i] for i in indices])
            
            auc_base = roc_auc_score(y_true_boot, y_pred_base_boot)
            auc_new = roc_auc_score(y_true_boot, y_pred_new_boot)
            delta_aucs.append(auc_new - auc_base)
            
            pk_base_scores = []
            pk_new_scores = []
            for i in indices:
                y = true_labels_list[i]
                K = int(y.sum())
                if K > 1:
                    top_k_base = np.argsort(pred_base_list[i])[-K:]
                    pk_base_scores.append(np.sum(y[top_k_base] == 1.0) / K)
                    
                    top_k_new = np.argsort(pred_new_list[i])[-K:]
                    pk_new_scores.append(np.sum(y[top_k_new] == 1.0) / K)
                    
            delta_pks.append(np.mean(pk_new_scores) - np.mean(pk_base_scores))
            
        ci_auc = (np.percentile(delta_aucs, 2.5), np.percentile(delta_aucs, 97.5))
        ci_pk = (np.percentile(delta_pks, 2.5), np.percentile(delta_pks, 97.5))
        
        print(f"\nDelta ROC-AUC (Hybrid - GNN): Mean={np.mean(delta_aucs):.4f}, 95% CI=({ci_auc[0]:.4f}, {ci_auc[1]:.4f})")
        print(f"Delta Precision@K (Hybrid - GNN): Mean={np.mean(delta_pks):.4f}, 95% CI=({ci_pk[0]:.4f}, {ci_pk[1]:.4f})")

    # Export to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "..", "evaluation_results.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "ROC-AUC", "Precision@K (Spread)", "Containment Accuracy", "Time MAE"])
        for name, metrics in results.items():
            cont_acc = metrics.get('Containment_Accuracy', 0.0)
            cont_str = f"{cont_acc:.4f}" if isinstance(cont_acc, float) else str(cont_acc)
            writer.writerow([
                name, 
                f"{metrics.get('ROCAUC', 0.0):.4f}", 
                f"{metrics.get('Precision_at_K', 0.0):.4f}", 
                cont_str, 
                metrics.get('Time_MAE', 'N/A')
            ])
            
    print(f"\nSaved evaluation summary to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers in CascadeNet")
    parser.add_argument('--use-supernode', action='store_true', help='Use supernode architecture')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size')
    args = parser.parse_args()
    
    run_all_evaluations(args)

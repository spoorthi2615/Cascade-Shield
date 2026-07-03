import os
import sys
import torch
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cascade_net import CascadeNet
from models.train import load_datasets
from scripts.run_evaluations import generate_static_topology

def diagnose():
    device = torch.device('cpu')
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    _, _, test_graphs = load_datasets(data_dir, smoke_test=False)
    
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    _, node_to_idx, idx_to_node = generate_static_topology()
    
    # Load model
    model = CascadeNet(in_channels=7, edge_dim=4, hidden_dim=64, num_layers=3, heads=4, use_supernode=False)
    hybrid_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model_hybrid.pth")
    
    if not os.path.exists(hybrid_path):
        print("Hybrid checkpoint not found!")
        return
        
    ckpt = torch.load(hybrid_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print("\n--- Diagnostic Check on 5 Test Graphs ---")
    for i, batch in enumerate(test_loader):
        if i >= 5:
            break
            
        with torch.no_grad():
            logits_clf, pred_time, edge_probs = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_edge_probs=True
            )
            
        N = batch.num_nodes
        origin_idx = torch.argmax(logits_clf[:N]).item()
        origin_name = idx_to_node[origin_idx]
        
        true_y = batch.y.numpy()
        valid_times = batch.y_time.clone()
        valid_times[batch.y_time < 0] = float('inf')
        true_origin_idx = torch.argmin(valid_times).item()
        true_origin_name = idx_to_node[true_origin_idx]
        
        infected_indices = np.where(true_y == 1.0)[0]
        infected_names = [idx_to_node[idx] for idx in infected_indices]
        
        print(f"\nGraph {i}:")
        print(f"  Batch size: num_graphs={batch.num_graphs}, num_nodes={N}")
        print(f"  True Origin: index={true_origin_idx}, name={true_origin_name}")
        print(f"  True Infected count: {len(infected_indices)}, nodes: {infected_names[:10]}")
        print(f"  GNN Predicted Origin: index={origin_idx}, name={origin_name}")
        print(f"  GNN Logit at Pred Origin: {logits_clf[origin_idx].item():.4f}")
        print(f"  GNN Logits Min: {logits_clf.min().item():.4f}, Max: {logits_clf.max().item():.4f}")
        print(f"  Is Pred Origin Infected? {true_y[origin_idx] == 1.0}")
        print(f"  Is Pred Origin True Origin? {origin_idx == true_origin_idx}")

if __name__ == "__main__":
    import numpy as np
    diagnose()

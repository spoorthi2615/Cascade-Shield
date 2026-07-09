import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train import load_datasets
from models.cascade_net import CascadeNet
from models.evaluate import evaluate
from baselines.mlp_baseline import MLPPredictor
from baselines.shared import BaselinePredictor

class DistanceHeuristic(BaselinePredictor):
    def __init__(self):
        super().__init__(name="DistanceHeuristic")
        self.is_temporal = False
    def predict(self, G, origin):
        # We will bypass the node lookup and just extract raw_weighted_dist from current_batch in evaluate.py
        dist = self.current_batch.raw_weighted_dist.numpy()
        dist[np.isinf(dist)] = dist[np.isfinite(dist)].max() + 1
        # In evaluate, score is used as probability, so smaller distance -> higher score
        max_dist = dist.max()
        if max_dist > 0:
            score = 1.0 - (dist / max_dist)
        else:
            score = np.ones_like(dist)
        
        # Build dict
        ret = {}
        for i in range(len(score)):
            name = self.idx_to_node[i]
            ret[name] = float(score[i])
        return ret

# We need an adapter for MLP because evaluate() passes (batch.x, batch.edge_index, batch.edge_attr, batch.batch)
class MLPWrapper(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
    def forward(self, x, edge_index, edge_attr, batch):
        logits = self.mlp(x)
        # return logits, dummy time
        return logits, torch.zeros_like(logits)

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data/processed/1000m'
    _, _, test_graphs = load_datasets(data_dir, smoke_test=False)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    print("\n--- GNN ---")
    gnn = CascadeNet(in_channels=test_graphs[0].x.shape[1], edge_dim=4, hidden_dim=64, num_layers=3, heads=4, use_supernode=False)
    gnn.load_state_dict(torch.load('checkpoints/width_64_seed_1/best_model.pth', map_location=device)['model_state_dict'])
    gnn.to(device).eval()
    res_gnn = evaluate(gnn, test_loader, device=device, is_baseline=False)
    print(f"GNN AUC: {res_gnn['ROCAUC']:.4f} | P@K: {res_gnn['Precision_at_K']:.4f}")
    
    print("\n--- MLP (Full) ---")
    mlp = MLPPredictor(in_channels=test_graphs[0].x.shape[1], hidden_dim=64)
    # We must train MLP quickly or load it. mlp_baseline train returns the model!
    from baselines.mlp_baseline import train_mlp
    _, _, best_mlp = train_mlp(data_dir, epochs=20, device=device)
    wrapped_mlp = MLPWrapper(best_mlp)
    wrapped_mlp.to(device).eval()
    res_mlp = evaluate(wrapped_mlp, test_loader, device=device, is_baseline=False)
    print(f"MLP AUC: {res_mlp['ROCAUC']:.4f} | P@K: {res_mlp['Precision_at_K']:.4f}")
    
    print("\n--- Distance Heuristic ---")
    dist_pred = DistanceHeuristic()
    # We need dummy G, idx_to_node, node_to_idx for evaluate to reconstruct the dict mapping
    import networkx as nx
    dist_pred.G = nx.Graph()
    dist_pred.idx_to_node = {i: str(i) for i in range(test_graphs[0].num_nodes)}
    dist_pred.node_to_idx = {str(i): i for i in range(test_graphs[0].num_nodes)}
    res_dist = evaluate(dist_pred, test_loader, device=device, is_baseline=True)
    print(f"Dist AUC: {res_dist['ROCAUC']:.4f} | P@K: {res_dist['Precision_at_K']:.4f}")

if __name__ == '__main__':
    run()

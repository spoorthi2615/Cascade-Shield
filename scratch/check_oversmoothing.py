import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader

sys.path.append(r'd:\projects\cascade sheild')
from models.cascade_net import CascadeNet
from models.train import load_datasets

def get_layer_cosine_similarities(model, graph, num_layers, use_supernode=False):
    model.eval()
    device = torch.device('cpu')
    graph = graph.to(device)
    
    sims = {}
    with torch.no_grad():
        x = model.node_proj(graph.x)
        x = F.relu(x)
        x_new, edge_index_new, edge_attr_new = x, graph.edge_index, graph.edge_attr
            
        x_0 = x_new
        for i in range(num_layers):
            x_res = x_new
            x_new = model.convs[i](x_new, edge_index_new, edge_attr_new)
            x_new = F.relu(x_new)
            x_new = (1 - model.alpha) * x_new + model.alpha * x_0
            if hasattr(model, 'norms'):
                x_new = model.norms[i](x_new)
            
            if (i + 1) in [1, 3, 5]:
                norms = torch.norm(x_new, p=2, dim=1, keepdim=True)
                norms = torch.clamp(norms, min=1e-8)
                normalized_x = x_new / norms
                
                similarity_matrix = torch.mm(normalized_x, normalized_x.t())
                N = similarity_matrix.size(0)
                indices = torch.triu_indices(N, N, offset=1)
                mean_sim = similarity_matrix[indices[0], indices[1]].mean().item()
                sims[i + 1] = mean_sim
                
        return sims

def main():
    data_dir = r'd:\projects\cascade sheild\data\processed'
    _, _, test_graphs = load_datasets(data_dir)
    in_channels = test_graphs[0].x.shape[1]
    
    # Load model
    model = CascadeNet(in_channels=in_channels, num_layers=5, use_supernode=False)
    ckpt = torch.load(r'd:\projects\cascade sheild\checkpoints\best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    
    print("\n--- Embedding Similarity Analysis (Oversmoothing Check) ---")
    
    layer_sims = {1: [], 3: [], 5: []}
    
    for g in test_graphs[:50]:
        sims = get_layer_cosine_similarities(model, g, num_layers=5, use_supernode=False)
        for layer, sim in sims.items():
            layer_sims[layer].append(sim)
            
    print("5-Layer Origin-Conditioned GNN with LayerNorm:")
    for layer in [1, 3, 5]:
        mean_sim = np.mean(layer_sims[layer])
        print(f"  Layer {layer} Mean Cosine Similarity: {mean_sim:.4f}")
        
    final_sim = np.mean(layer_sims[5])
    if final_sim >= 0.86:
        print(f"\n[WARNING] Oversmoothing detected! Final Similarity {final_sim:.4f} >= threshold 0.86")
    else:
        print(f"\n[PASS] Model avoided oversmoothing. Final Similarity {final_sim:.4f} < threshold 0.86")

if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F
import sys
import numpy as np

sys.path.append(r'd:\projects\cascade sheild')
from models.train import load_datasets
from models.cascade_net import CascadeNet

def main():
    data_dir = r'd:\projects\cascade sheild\data\processed'
    _, _, test_graphs = load_datasets(data_dir, smoke_test=False)
    
    # We just need the raw features of one graph, e.g., the first test graph
    g = test_graphs[0]
    
    # Feature vector layout from user's prompt:
    # [criticality, power, water, traffic, transit, in_degree, out_degree]
    # (actually 8 dims in Phase 2 with origin conditioning, but the subsystem dims are usually indices 1-4)
    # Let's just find nodes belonging to each subsystem and check similarities
    
    # Instantiate node_proj to get the projection
    model = CascadeNet(in_channels=g.x.shape[1], use_supernode=False)
    model.eval()
    
    with torch.no_grad():
        x_proj = model.node_proj(g.x)
        x_proj = F.relu(x_proj)
        
        # Normalize
        norms = torch.norm(x_proj, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        x_norm = x_proj / norms
        
        # We need to identify subsystems. Assuming dims 1,2,3,4 (0-indexed) are the one-hot subsystem flags
        # Let's inspect the raw features to see where the one-hot vectors are.
        # Print a few raw rows:
        print("Raw features shape:", g.x.shape)
        
        # Just compute all pairs and group by whether they share the exact same one-hot signature
        # A node's subsystem signature is roughly its values in dims 1..4 (or 1..n if different)
        # Let's just use the raw features to group them
        sim_matrix = torch.mm(x_norm, x_norm.t())
        
        N = g.x.shape[0]
        within_sims = []
        cross_sims = []
        
        for i in range(N):
            for j in range(i + 1, N):
                # We consider them same subsystem if their binary flags match exactly.
                # Assuming the middle columns (e.g. 1:5) are the one-hots. We can just check if g.x[i] and g.x[j] 
                # share the highest value in the subsystem slice. Or simpler: 
                # do they have identical values for the one-hot slice?
                # Actually, let's just group by exact match on the one-hot slice.
                # Let's guess the one-hot slice is dims 1 to 5 based on "power, water, traffic, transit" (4 dims)
                
                slice_i = g.x[i, 1:5]
                slice_j = g.x[j, 1:5]
                is_same = torch.allclose(slice_i, slice_j)
                
                sim = sim_matrix[i, j].item()
                if is_same:
                    within_sims.append(sim)
                else:
                    cross_sims.append(sim)
                    
        print(f"Mean Within-Subsystem Similarity:  {np.mean(within_sims):.4f} (from {len(within_sims)} pairs)")
        print(f"Mean Cross-Subsystem Similarity:   {np.mean(cross_sims):.4f} (from {len(cross_sims)} pairs)")

if __name__ == "__main__":
    main()

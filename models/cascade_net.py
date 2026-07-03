import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class CascadeNet(nn.Module):
    def __init__(self, in_channels: int = 9, edge_dim: int = 4, hidden_dim: int = 64, num_layers: int = 3, heads: int = 4, use_supernode: bool = False):
        super(CascadeNet, self).__init__()
        
        self.num_layers = num_layers
        self.use_supernode = use_supernode
        
        # We use a linear projection to map input features to the hidden dimension
        self.node_proj = nn.Linear(in_channels, hidden_dim)
        
        # GATv2 layers (using multiple heads, concatenating the output so internal dim is hidden_dim // heads)
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        head_dim = hidden_dim // heads
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim, 
                    out_channels=head_dim, 
                    heads=heads, 
                    edge_dim=edge_dim, 
                    concat=True
                )
            )
            
        # Supernode parameters
        if use_supernode:
            self.supernode_emb = nn.Parameter(torch.randn(1, hidden_dim))
            # Directional virtual edge attributes
            self.virtual_edge_attr_in = nn.Parameter(torch.randn(1, edge_dim))
            self.virtual_edge_attr_out = nn.Parameter(torch.randn(1, edge_dim))
            
        # Dual Output Heads
        # Classification Head: predicts probability of infection/failure [0.0 - 1.0]
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
            # We omit the Sigmoid here because PyTorch's BCEWithLogitsLoss / FocalLoss
            # is numerically more stable when applied to raw logits.
        )
        
        # Regression Head: predicts time-to-impact (>= 0)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus() # Smooth, positive activation that never dies
        )
        
        # Edge Head: predicts transmission probability for topological edges
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # Edge probabilities in [0.0, 1.0]
        )

    def forward(self, x, edge_index, edge_attr, batch_idx=None, return_edge_probs=False):
        # 1. Project node features
        x = self.node_proj(x)
        x = F.relu(x)
        
        N = x.size(0)
        
        # 2. Append batch-aware supernodes
        if self.use_supernode:
            if batch_idx is None:
                batch_idx = torch.zeros(N, dtype=torch.long, device=x.device)
            B = batch_idx.max().item() + 1
            
            # Append learned supernodes features
            super_x = self.supernode_emb.expand(B, -1)
            x_new = torch.cat([x, super_x], dim=0)
            
            # Connect real nodes to their virtual supernode bidirectionally
            super_indices = torch.arange(N, N + B, device=x.device)
            col = super_indices[batch_idx] # virtual indices corresponding to each real node
            row = torch.arange(N, device=x.device) # real indices
            
            # Edges: real -> virtual
            v_edges_in = torch.stack([row, col], dim=0)
            # Edges: virtual -> real
            v_edges_out = torch.stack([col, row], dim=0)
            edge_index_new = torch.cat([edge_index, v_edges_in, v_edges_out], dim=1)
            
            # Directional virtual edge attributes
            v_attrs_in = self.virtual_edge_attr_in.expand(N, -1)
            v_attrs_out = self.virtual_edge_attr_out.expand(N, -1)
            edge_attr_new = torch.cat([edge_attr, v_attrs_in, v_attrs_out], dim=0)
        else:
            x_new, edge_index_new, edge_attr_new = x, edge_index, edge_attr
        
        # 3. Message Passing
        for i in range(self.num_layers):
            x_res = x_new
            x_new = self.convs[i](x_new, edge_index_new, edge_attr_new)
            x_new = F.relu(x_new)
            # Add residual connection to prevent oversmoothing
            x_new = x_new + x_res
            
        # 4. Dual Heads
        logits_clf = self.clf_head(x_new).squeeze(-1) # [num_nodes + B]
        pred_time = self.reg_head(x_new).squeeze(-1)  # [num_nodes + B]
        
        # Slice out supernodes to return only real node predictions
        if self.use_supernode:
            logits_clf = logits_clf[:N]
            pred_time = pred_time[:N]
            
        if return_edge_probs:
            # Predict only for original topological edges (edge_index)
            # Fetch representations of source and target nodes
            h_u = x_new[edge_index[0]]
            h_v = x_new[edge_index[1]]
            edge_feats = torch.cat([h_u, h_v], dim=1)
            edge_probs = self.edge_head(edge_feats).squeeze(-1)
            return logits_clf, pred_time, edge_probs
            
        return logits_clf, pred_time

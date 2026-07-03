import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from models.cascade_net import CascadeNet
from models.evaluate import evaluate
from models.train import load_datasets

device = torch.device('cpu')
data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
_, _, test_graphs = load_datasets(data_dir, smoke_test=False)
test_loader = DataLoader(test_graphs, batch_size=4, shuffle=False)

def eval_model(hidden_dim, seed):
    ckpt_path = os.path.join("checkpoints", f"width_{hidden_dim}_seed_{seed}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Missing {ckpt_path}")
        return None
    model = CascadeNet(in_channels=9, edge_dim=4, hidden_dim=hidden_dim, num_layers=3, heads=4, use_supernode=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    res = evaluate(model, test_loader, device=device, is_baseline=False)
    return res['ROCAUC']

for dim in [64, 128]:
    aucs = []
    for seed in [0, 1, 2]:
        auc = eval_model(dim, seed)
        if auc is not None:
            aucs.append(auc)
            print(f"Dim {dim} Seed {seed}: {auc:.4f}")
    if aucs:
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"Dim {dim} Summary: {mean_auc:.4f} ± {std_auc:.4f}\n")

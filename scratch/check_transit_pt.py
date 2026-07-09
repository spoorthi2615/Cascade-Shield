import torch
import os

pt_path = os.path.join("data", "processed", "1000m", "scenario_0.pt")
if os.path.exists(pt_path):
    g = torch.load(pt_path, weights_only=False)
    transit_count = g.x[:, 4].sum().item()
    print("Transit nodes in PT:", transit_count)
else:
    print("PT file not found")

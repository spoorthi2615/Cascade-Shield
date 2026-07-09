import torch, glob, numpy as np

files = glob.glob('data/processed/1000m/*.pt')
print(f"Found {len(files)} .pt files")
if files:
    rates = []
    for f in files[:30]:
        data = torch.load(f, weights_only=False)
        rates.append(data.y.mean().item())
    print(f"Mean positive rate in old PTs: {np.mean(rates):.4f}")

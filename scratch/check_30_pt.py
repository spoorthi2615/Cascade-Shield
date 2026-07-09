import torch, glob, numpy as np
files = sorted(glob.glob('data/processed/1000m/*.pt'))
if files:
    train_g, val_g, test_g = [], [], []
    for f in files:
        data = torch.load(f, weights_only=False)
        if data.split == 'test':
            test_g.append(data.y.mean().item())
    
    test_g_30 = test_g[:30]
    print(f"First 30 test graphs pos rate: {np.mean(test_g_30):.4f}")
    print(f"All test graphs pos rate: {np.mean(test_g):.4f}")

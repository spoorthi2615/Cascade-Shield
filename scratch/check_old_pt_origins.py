import torch, glob
files = sorted(glob.glob('data/processed/1000m/*.pt'))
origins = []
for f in files[:300]:
    data = torch.load(f, weights_only=False)
    valid_times = data.y_time.clone()
    valid_times[data.y_time < 0] = float('inf')
    origin_idx = torch.argmin(valid_times).item()
    origins.append(origin_idx)
    
pwr, water, road = 0, 0, 0
for idx in origins:
    if idx < 128: pwr += 1
    elif idx < 128+99: water += 1
    else: road += 1
print(f"Power: {pwr}, Water: {water}, Road: {road}")

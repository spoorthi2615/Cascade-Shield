import json, numpy as np
def check(file):
    try:
        with open(file) as f: d = json.load(f)
        sizes = [len(s['cascade_path'])/2371 for s in d]
        print(f"{file}:\n  Scenarios: {len(sizes)}\n  Median frac: {np.median(sizes):.4f}\n  Mean frac: {np.mean(sizes):.4f}")
    except Exception as e: print(e)
check('data/simulator/dataset/1000m/train.json')
check('data/simulator/dataset/1000m/test.json')

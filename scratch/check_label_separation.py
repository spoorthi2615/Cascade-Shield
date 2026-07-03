import torch
import sys
import numpy as np

sys.path.append(r'd:\projects\cascade sheild')
from models.train import load_datasets

def main():
    data_dir = r'd:\projects\cascade sheild\data\processed'
    train, val, test = load_datasets(data_dir, smoke_test=False)
    
    all_graphs = train + val + test
    
    # We want to see if y_clf == 1 is perfectly predictable by distance
    all_dists = []
    all_labels = []
    
    for g in all_graphs:
        # Distance is the 9th feature (index 8)
        dist = g.x[:, 8].numpy()
        label = g.y.numpy()
        
        # Don't include the origin itself in the evaluation of "predictability" 
        # (it has distance 0 and label 1, but we know it's always infected)
        valid_times = g.y_time.clone()
        valid_times[g.y_time < 0] = float('inf')
        origin_idx = torch.argmin(valid_times).item()
        
        mask = np.ones(len(dist), dtype=bool)
        mask[origin_idx] = False
        
        all_dists.extend(dist[mask])
        all_labels.extend(label[mask])
        
    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)
    
    print(f"Total evaluated nodes (excluding origins): {len(all_labels)}")
    print(f"Total positive (infected): {np.sum(all_labels == 1)}")
    print(f"Total negative (safe): {np.sum(all_labels == 0)}")
    
    print("\n--- Distribution of Distances ---")
    pos_dists = all_dists[all_labels == 1]
    neg_dists = all_dists[all_labels == 0]
    
    print(f"Positive nodes distance: Mean={np.mean(pos_dists):.4f}, Min={np.min(pos_dists):.4f}, Max={np.max(pos_dists):.4f}")
    if len(neg_dists) > 0:
        print(f"Negative nodes distance: Mean={np.mean(neg_dists):.4f}, Min={np.min(neg_dists):.4f}, Max={np.max(neg_dists):.4f}")
    
    # Let's try to find a threshold that separates them best
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(all_labels, -all_dists) # smaller distance = more likely
    print(f"\nGlobal ROC-AUC of raw -distance alone: {auc:.4f}")
    
    fpr, tpr, thresholds = roc_curve(all_labels, -all_dists)
    # Youden's J statistic
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = -thresholds[best_idx]
    
    print(f"Optimal distance threshold for separation: {best_thresh:.4f}")
    
    # Check fractions at this threshold
    below_thresh_pos = np.sum((all_dists <= best_thresh) & (all_labels == 1))
    below_thresh_neg = np.sum((all_dists <= best_thresh) & (all_labels == 0))
    above_thresh_pos = np.sum((all_dists > best_thresh) & (all_labels == 1))
    above_thresh_neg = np.sum((all_dists > best_thresh) & (all_labels == 0))
    
    print(f"\nNodes <= {best_thresh:.4f} distance:")
    print(f"  Positive: {below_thresh_pos}")
    print(f"  Negative: {below_thresh_neg}")
    print(f"Nodes > {best_thresh:.4f} distance:")
    print(f"  Positive: {above_thresh_pos}")
    print(f"  Negative: {above_thresh_neg}")

if __name__ == "__main__":
    main()

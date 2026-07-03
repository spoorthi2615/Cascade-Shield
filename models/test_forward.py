import os
import sys
import torch
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cascade_net import CascadeNet
from models.loss import FocalLoss, masked_mse_loss

def run_forward_test():
    print("--- Running Forward Pass Verification ---")
    
    # 1. Load a real serialized scenario
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "scenario_0.pt")
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        sys.exit(1)
        
    # We must use weights_only=False because PyG Data objects are loaded
    sample = torch.load(file_path, weights_only=False)
    num_nodes = sample.x.shape[0]
    
    # 2. Instantiate CascadeNet
    model = CascadeNet(in_channels=7, edge_dim=4, hidden_dim=64, num_layers=3, heads=4)
    model.eval()
    
    # 3. Run Forward Pass
    logits_clf, pred_time = model(sample.x, sample.edge_index, sample.edge_attr)
    
    # Assertions on Shapes
    assert logits_clf.shape == (num_nodes,), f"Expected logits shape {(num_nodes,)}, got {logits_clf.shape}"
    assert pred_time.shape == (num_nodes,), f"Expected pred_time shape {(num_nodes,)}, got {pred_time.shape}"
    print("[PASS] Forward pass shape verification: logits and pred_time both match [num_nodes]")
    
    # 4. Loss calculation tests
    focal_criterion = FocalLoss()
    clf_loss = focal_criterion(logits_clf, sample.y)
    
    assert not torch.isnan(clf_loss), "Classification loss returned NaN"
    print(f"[PASS] Classification Loss (Focal) computed successfully: {clf_loss.item():.4f}")
    
    reg_loss = masked_mse_loss(pred_time, sample.y_time, sample.y)
    assert not torch.isnan(reg_loss), "Regression loss returned NaN"
    print(f"[PASS] Regression Loss (Masked MSE) computed successfully: {reg_loss.item():.4f}")
    
    # 5. Edge Case Test: What if the cascade is size 0 (mask sum is 0)?
    dummy_pred_time = torch.ones_like(pred_time)
    dummy_true_time = torch.ones_like(pred_time)
    dummy_y_zeros = torch.zeros_like(sample.y) # strictly all zeros
    
    reg_loss_zeros = masked_mse_loss(dummy_pred_time, dummy_true_time, dummy_y_zeros)
    assert not torch.isnan(reg_loss_zeros), "Regression loss returned NaN on all-zero mask"
    assert reg_loss_zeros.item() == 0.0, "Regression loss should be 0.0 when mask is empty"
    print("[PASS] Edge Case: Masked MSE handles empty mask (0 infected nodes) without NaN")
    
    print("All architecture verifications passed!")

if __name__ == "__main__":
    run_forward_test()

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cascade_net import CascadeNet
import torch.optim as optim

def test_checkpoint():
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "checkpoint_latest.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
        
    print(f"Loading checkpoint from {ckpt_path}...")
    
    # 1. Instantiate fresh model and optimizer
    device = torch.device("cpu")
    model = CascadeNet(in_channels=7, edge_dim=4, hidden_dim=64, num_layers=3, heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 2. Load state
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    best_val = checkpoint['best_val_loss']
    
    print(f"[PASS] Model and Optimizer state dictionaries loaded successfully!")
    print(f"[PASS] Resumed at Epoch: {epoch}")
    print(f"[PASS] Best Val Loss recorded: {best_val:.4f}")
    
    # 3. Quick forward pass to ensure weights aren't corrupt
    dummy_x = torch.zeros((10, 7))
    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
    dummy_edge_attr = torch.zeros((0, 4))
    
    model.eval()
    with torch.no_grad():
        out_clf, out_time = model(dummy_x, dummy_edge_index, dummy_edge_attr)
        
    assert out_clf.shape == (10,)
    assert out_time.shape == (10,)
    print("[PASS] Forward pass using loaded weights succeeded without error.")

if __name__ == "__main__":
    test_checkpoint()

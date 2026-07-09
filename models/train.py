import os
import sys
import glob
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cascade_net import CascadeNet
from models.loss import FocalLoss, masked_mse_loss
from models.evaluate import evaluate

import scipy.sparse as sp
import numpy as np
import argparse
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def add_distance_feature(g):
    N = g.x.shape[0]
    edge_index = g.edge_index.cpu().numpy()
    weights = g.edge_attr[:, 3].cpu().numpy()
    costs = 1.0 / (weights + 1e-5)
    
    adj = sp.csr_matrix((costs, (edge_index[0], edge_index[1])), shape=(N, N))
    
    valid_times = g.y_time.clone()
    valid_times[g.y_time < 0] = float('inf')
    origin_idx = torch.argmin(valid_times).item()
    
    dist, predecessors = sp.csgraph.shortest_path(adj, directed=True, indices=origin_idx, return_predecessors=True)
    
    # Handle unreachable nodes
    max_finite = np.max(dist[np.isfinite(dist)])
    dist[np.isinf(dist)] = max_finite
    
    # Normalize min-max
    d_min = dist.min()
    d_max = dist.max()
    if d_max > d_min:
        dist_norm = (dist - d_min) / (d_max - d_min)
    else:
        dist_norm = np.ones_like(dist)
        
    assert not np.any(np.isnan(dist_norm)), "NaN in distance feature"
    assert not np.any(np.isinf(dist_norm)), "Inf in distance feature"
    
    dist_tensor = torch.tensor(dist_norm, dtype=torch.float32).unsqueeze(1).to(g.x.device)
    
    if not getattr(g, 'ablate_distance', False):
        g.x = torch.cat([g.x, dist_tensor], dim=1)
    
    # Also save raw distance for naive heuristic baseline
    g.raw_weighted_dist = torch.tensor(dist, dtype=torch.float32)
    return g

def load_datasets(data_dir: str, smoke_test: bool = False, ablate_distance: bool = False):
    """
    Loads serialized PyG graphs and partitions them by the 'split' attribute.
    If smoke_test is true, loads only 5 train and 2 val graphs.
    """
    files = glob.glob(os.path.join(data_dir, "scenario_*.pt"))
    if not files:
        raise ValueError(f"No scenario files found in {data_dir}")
        
    train_graphs, val_graphs, test_graphs = [], [], []
    
    # Sort files to ensure deterministic loading across runs
    files = sorted(files)
    
    for f in files:
        try:
            # Must use weights_only=False for custom PyG objects
            g = torch.load(f, weights_only=False)
            g.ablate_distance = ablate_distance
            g = add_distance_feature(g)
            if g.split == "train":
                train_graphs.append(g)
            elif g.split == "val":
                val_graphs.append(g)
            elif g.split == "test":
                test_graphs.append(g)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            
    if smoke_test:
        train_graphs = train_graphs[:100]
        val_graphs = val_graphs[:20]
        test_graphs = []
        print(f"--- SMOKE TEST MODE: Loaded {len(train_graphs)} Train, {len(val_graphs)} Val ---")
    else:
        print(f"--- Loaded {len(train_graphs)} Train, {len(val_graphs)} Val, {len(test_graphs)} Test ---")
        
    return train_graphs, val_graphs, test_graphs

def hard_negative_ranking_loss(logits, targets, batch_index, margin=0.1):
    """
    For each graph in the batch, and for each positive node in that graph,
    pairs it with the single highest-scoring negative node in that same graph.
    Computes MarginRankingLoss.
    """
    logits = logits.squeeze()
    loss = 0.0
    num_pairs = 0
    
    for b in range(batch_index.max().item() + 1):
        mask = (batch_index == b)
        b_logits = logits[mask]
        b_targets = targets[mask]
        
        pos_mask = (b_targets == 1.0)
        neg_mask = (b_targets == 0.0)
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
            
        pos_logits = b_logits[pos_mask]
        neg_logits = b_logits[neg_mask]
        
        # Hard negative mining: Find the max logit among negatives
        hard_neg_logit = neg_logits.max()
        
        x2 = hard_neg_logit.expand_as(pos_logits)
        y = torch.ones_like(pos_logits)
        
        l = F.margin_ranking_loss(pos_logits, x2, y, margin=margin, reduction='sum')
        loss += l
        num_pairs += len(pos_logits)
        
    if num_pairs > 0:
        return loss / num_pairs
    return torch.tensor(0.0, device=logits.device, requires_grad=True)

def get_node_weights(batch, is_train=True):
    # 1. Positives
    pos_mask = batch.y == 1.0
    
    # 2. Boundary (1-Hop)
    # Find edges originating from an infected node
    infected_src = pos_mask[batch.edge_index[0]]
    boundary_nodes = batch.edge_index[1][infected_src]
    
    boundary_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
    boundary_mask[boundary_nodes] = True
    # Exclude nodes that are already positive
    boundary_mask = boundary_mask & (~pos_mask)
    
    # 3. Background
    background_mask = ~(pos_mask | boundary_mask)
    
    node_weights = torch.zeros_like(batch.y)
    node_weights[pos_mask] = 1.0
    node_weights[boundary_mask] = 2.0 # Double weight on the boundary zone
    node_weights[background_mask] = 1.0 # Keep background at full weight
        
    return node_weights, pos_mask.sum().item(), boundary_mask.sum().item(), background_mask.sum().item()

def run_training(args):
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Kaggle vs Local paths
        if os.path.exists("/kaggle/working/"):
            base_dir = "/kaggle/working/"
            data_dir = "/kaggle/input/cascade-shield-data/data/processed"
            if not os.path.exists(data_dir):
                data_dir = os.path.join(base_dir, "data", "processed")
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data", "processed", "1000m") # Default to 1000m
            
    if hasattr(args, 'save_dir') and args.save_dir:
        ckpt_dir = args.save_dir
    else:
        seed_str = f"seed_{args.seed}" if getattr(args, 'seed', None) is not None else "seed_None"
        hidden_dim = getattr(args, 'hidden_dim', 64)
        ckpt_dir = os.path.join(base_dir if 'base_dir' in locals() else os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", f"width_{hidden_dim}_{seed_str}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_graphs, val_graphs, _ = load_datasets(data_dir, smoke_test=args.smoke_test, ablate_distance=getattr(args, 'ablate_distance', False))
    
    in_channels = train_graphs[0].x.shape[1]
    expected_dim = 8 if getattr(args, 'ablate_distance', False) else 9
    assert in_channels == expected_dim, f"Expected {expected_dim}-dim features, but got {in_channels}"
    
    # Batch size 4 prevents OOM on 4GB VRAM while still providing batch stability
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    
    hidden_dim = getattr(args, 'hidden_dim', 64)
    model = CascadeNet(in_channels=in_channels, edge_dim=4, hidden_dim=hidden_dim, num_layers=args.num_layers, 
                       heads=4, use_supernode=args.use_supernode).to(device)
    learning_rate = getattr(args, 'lr', 0.001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    focal_criterion = FocalLoss(gamma=2.0, alpha=0.25).to(device)
    
    # Lambda importance weighting for the dual-head loss
    # time_loss is normalized to [0,1], but we still consider infection the primary task.
    LAMBDA_TIME = 0.5 
    MAX_TICK_HORIZON = 50.0
    LAMBDA_RANK = getattr(args, 'lambda_rank', 0.0)
    
    epochs = 20 if args.smoke_test else args.epochs
    
    best_val_auc = 0.0
    first_epoch_loss = None
    start_epoch = 1
    
    if getattr(args, 'resume', False):
        resume_path = os.path.join(ckpt_dir, "checkpoint_latest.pth")
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_auc = ckpt['val_auc']
            print(f"Resuming from epoch {start_epoch-1} with Val AUC {best_val_auc:.4f}")
    
    print("\nStarting Training Loop...")
    for epoch in range(start_epoch, epochs + 1):
        # --- TRAINING ---
        model.train()
        train_clf_tot, train_time_tot, train_rank_tot, train_tot = 0.0, 0.0, 0.0, 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if getattr(args, 'train_hybrid', False):
                logits_clf, pred_time, edge_probs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_edge_probs=True)
                
                # Reconstruct ground-truth edge labels on adjacent infected nodes
                u = batch.edge_index[0]
                v = batch.edge_index[1]
                y_u = batch.y[u]
                y_v = batch.y[v]
                t_u = batch.y_time[u]
                t_v = batch.y_time[v]
                
                y_edge = ((y_u == 1.0) & (y_v == 1.0) & (t_u >= 0) & (t_u < t_v)).float()
                
                edge_probs = torch.clamp(edge_probs, min=1e-7, max=1-1e-7)
                edge_loss = F.binary_cross_entropy(edge_probs, y_edge)
                
                clf_loss = focal_criterion(logits_clf, batch.y)
                norm_y_time = batch.y_time / 50.0
                time_loss = masked_mse_loss(pred_time, norm_y_time, batch.y)
                
                # Decoupled losses combined (equal weight to edge capacity prediction)
                total_loss = clf_loss + (LAMBDA_TIME * time_loss) + edge_loss
                train_rank_tot += 0.0
            else:
                logits_clf, pred_time = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                clf_loss = focal_criterion(logits_clf, batch.y)
                norm_y_time = batch.y_time / 50.0
                time_loss = masked_mse_loss(pred_time, norm_y_time, batch.y)
                rank_loss = hard_negative_ranking_loss(logits_clf, batch.y, batch.batch, margin=0.1)
                total_loss = clf_loss + (LAMBDA_TIME * time_loss) + (LAMBDA_RANK * rank_loss)
                train_rank_tot += rank_loss.item()
            
            total_loss.backward()
            optimizer.step()
            
            train_clf_tot += clf_loss.item()
            train_time_tot += time_loss.item()
            train_tot += total_loss.item()
            
        n_train = len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_clf_tot, val_time_tot, val_rank_tot, val_tot = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits_clf, pred_time = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                clf_loss = focal_criterion(logits_clf, batch.y)
                norm_y_time = batch.y_time / MAX_TICK_HORIZON
                time_loss = masked_mse_loss(pred_time, norm_y_time, batch.y)
                rank_loss = hard_negative_ranking_loss(logits_clf, batch.y, batch.batch, margin=0.1)
                total_loss = clf_loss + (LAMBDA_TIME * time_loss) + (LAMBDA_RANK * rank_loss)
                
                val_clf_tot += clf_loss.item()
                val_time_tot += time_loss.item()
                val_rank_tot += rank_loss.item()
                val_tot += total_loss.item()
                
        n_val = len(val_loader)
        
        # Averages
        avg_train_tot = train_tot / n_train
        avg_val_tot = val_tot / n_val if n_val > 0 else 0.0
        
        # Calculate Validation metrics using unified evaluate harness
        val_metrics = evaluate(model, val_loader, device=device)
        val_auc = val_metrics['ROCAUC']
        val_p_k = val_metrics['Precision_at_K']
        
        if epoch == 1:
            first_epoch_loss = avg_train_tot
            
        print(f"Epoch {epoch:03d} | "
              f"Train Tot: {avg_train_tot:.4f} (Clf: {train_clf_tot/n_train:.4f}, Rank: {train_rank_tot/n_train:.4f}) | "
              f"Val Tot: {avg_val_tot:.4f} (Clf: {val_clf_tot/n_val:.4f}, Rank: {val_rank_tot/n_val:.4f}) | "
              f"Val AUC: {val_auc:.4f} | Val P@K: {val_p_k:.4f}")
              
        # ckpt_dir already set at the start of run_training
        pass
        
        # --- CHECKPOINTING ---
        ckpt_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc
        }
        
        # Always save latest for resumption
        torch.save(ckpt_state, os.path.join(ckpt_dir, "checkpoint_latest.pth"))
        
        # Keep track of top 3 models by Val AUC
        if not hasattr(model, 'top_k_checkpoints'):
            model.top_k_checkpoints = []
            
        if n_val > 0:
            filename = f"best_model_epoch_{epoch:03d}.pth" if not getattr(args, 'train_hybrid', False) else f"best_hybrid_epoch_{epoch:03d}.pth"
            filepath = os.path.join(ckpt_dir, filename)
            
            # Add to list and sort
            model.top_k_checkpoints.append((val_auc, filepath, ckpt_state))
            model.top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)
            
            # If we just added one and we are still within top 3, or if we beat the 4th, save it
            # Actually, it's easier to just write out the top 3 and delete any that fall off
            
            # Ensure we only keep 3
            while len(model.top_k_checkpoints) > 3:
                _, drop_path, _ = model.top_k_checkpoints.pop()
                if os.path.exists(drop_path):
                    os.remove(drop_path)
                    
            # Save all current top K to their epoch-specific filenames
            for _, path, state in model.top_k_checkpoints:
                if not os.path.exists(path):
                    torch.save(state, path)
                    
            # And also symlink or just save the absolute best as 'best_model.pth' for easy eval compatibility
            best_filename = "best_model_hybrid.pth" if getattr(args, 'train_hybrid', False) else "best_model.pth"
            torch.save(model.top_k_checkpoints[0][2], os.path.join(ckpt_dir, best_filename))
            
    # --- SMOKE TEST VALIDATION ---
    if args.smoke_test:
        final_epoch_loss = avg_train_tot
        if final_epoch_loss >= first_epoch_loss:
            print(f"\n[FATAL] Smoke Test Failed! Loss did not decrease. (Initial: {first_epoch_loss:.4f}, Final: {final_epoch_loss:.4f})")
            sys.exit(1)
        else:
            print(f"\n[PASS] Smoke Test Successful! Loss decreased from {first_epoch_loss:.4f} to {final_epoch_loss:.4f}.")
            
    # --- FINAL TEST EVALUATION ---
    print("\n--- Running Final Evaluation on Test Set ---")
    _, _, test_graphs = load_datasets(data_dir, smoke_test=args.smoke_test, ablate_distance=getattr(args, 'ablate_distance', False))
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    # Load the best model
    best_filename = "best_model_hybrid.pth" if getattr(args, 'train_hybrid', False) else "best_model.pth"
    best_path = os.path.join(ckpt_dir, best_filename)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
        print(f"Loaded best model from {best_path}")
    
    model.eval()
    test_metrics = evaluate(model, test_loader, device=device)
    print(f"FINAL TEST AUC: {test_metrics['ROCAUC']:.4f}")
    print(f"FINAL TEST P@K: {test_metrics['Precision_at_K']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 5-epoch test on 5 samples to verify backward pass.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of GNN message passing layers")
    parser.add_argument('--use-supernode', action='store_true', help='Enable supernode architecture')
    parser.add_argument('--data-dir', type=str, default=None, help='Override dataset directory')
    parser.add_argument('--train-hybrid', action='store_true', help='Train Phase 1 Hybrid (Edge prediction)')
    parser.add_argument('--lambda-rank', type=float, default=0.0, help='Weight for pairwise ranking loss')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--ablate-distance', action='store_true', help='Remove distance feature (Feature 8) for ablation')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-dir', type=str, default=None, help='Specific directory to save checkpoints')
    args = parser.parse_args()
    
    if args.seed is not None:
        set_seed(args.seed)
    
    run_training(args)

import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def evaluate(model_or_baseline, test_loader, device=None, is_baseline=False):
    """
    Unified Evaluation Harness for Cascade Shield.
    
    Accepts either:
    - CascadeNet (PyTorch model)
    - BaselinePredictor (e.g. ClassicalSEIR, StaticCentrality)
    
    Returns:
    dict: {
        'ROCAUC': float,
        'Precision_at_K': float,
        'Containment_Accuracy': float,
        'Time_MAE': float or "N/A (non-temporal)"
    }
    """
    if not is_baseline and device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_or_baseline.eval()
        
    all_y_true = []
    all_y_pred = []
    
    # Per-scenario tracking for Precision@K and Containment
    precision_scores = []
    containment_correct = 0
    containment_total = 0
    
    # MAE tracking
    mae_diffs = []
    temporal_supported = True
    
    # Evaluate across all scenarios in the loader
    for batch in test_loader:
        if is_baseline:
            # BASELINE PREDICTOR
            model_or_baseline.current_batch = batch
            # 1. Infer origin index from y_time (earliest infected node)
            valid_times = batch.y_time.clone()
            valid_times[batch.y_time < 0] = float('inf')
            origin_idx = torch.argmin(valid_times).item()
            
            # Map index back to string name if needed
            # For this we need the static networkx graph and idx mapping
            # Assuming model_or_baseline has a reference to them or we pass them
            origin_name = model_or_baseline.idx_to_node[origin_idx]
            
            # Call predict (returns dict of node -> continuous risk [0,1])
            # And potentially time prediction if supported
            # BaselinePredictor contract: predict(G, origin) -> dict
            risk_dict = model_or_baseline.predict(model_or_baseline.G, origin_name)
            
            # Reconstruct y_pred tensor to match the batch indexing
            pred_prob = torch.zeros(batch.num_nodes, dtype=torch.float)
            for name, score in risk_dict.items():
                idx = model_or_baseline.node_to_idx[name]
                pred_prob[idx] = score
                
            y_pred = pred_prob.numpy()
            
            if getattr(model_or_baseline, 'is_temporal', False):
                # If baseline supports temporal prediction, fetch it
                time_dict = model_or_baseline.predict_time(model_or_baseline.G, origin_name)
                pred_time = torch.zeros(batch.num_nodes, dtype=torch.float)
                for name, t in time_dict.items():
                    idx = model_or_baseline.node_to_idx[name]
                    pred_time[idx] = t
                pred_t = pred_time.numpy()
                temporal_supported = True
            else:
                pred_t = None
                temporal_supported = False
                
            y_true = batch.y.numpy()
            y_time = batch.y_time.numpy()
            
        else:
            # PyTorch Model
            batch = batch.to(device)
            with torch.no_grad():
                logits_clf, pred_time_norm = model_or_baseline(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred_prob = torch.sigmoid(logits_clf)
                pred_time = pred_time_norm * 50.0 # Denormalize back to ticks
                
            y_true = batch.y.cpu().numpy()
            y_pred = pred_prob.cpu().numpy()
            y_time = batch.y_time.cpu().numpy()
            pred_t = pred_time.cpu().numpy()
            temporal_supported = True
            
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        infected_mask = (y_true == 1.0)
        K = int(np.sum(infected_mask))
            
        # --- 1. Containment vs Spread (Precision@K) ---
        if K <= 1:
            containment_total += 1
            # If contained, we consider a correct prediction if all other nodes score < 0.5
            if np.all(y_pred[y_true == 0.0] < 0.5):
                containment_correct += 1
        else:
            # K > 1: Calculate Precision@K
            # Find indices of top K predictions
            top_k_indices = np.argsort(y_pred)[-K:]
            # How many of the top K are actually infected?
            true_positives = np.sum(infected_mask[top_k_indices])
            precision_scores.append(true_positives / K)
            
        # --- 2. Time MAE ---
        if K > 1 and temporal_supported:
            mae_diffs.extend(np.abs(pred_t[infected_mask] - y_time[infected_mask]))
                
    # Compile Metrics
    metrics = {}
    
    # ROC-AUC
    try:
        metrics['ROCAUC'] = roc_auc_score(all_y_true, all_y_pred)
    except ValueError:
        metrics['ROCAUC'] = 0.5 # Default if only one class exists in entire test set
        
    # Spread Quality
    metrics['Precision_at_K'] = np.mean(precision_scores) if precision_scores else 0.0
    
    # Containment Quality
    metrics['Containment_Accuracy'] = (containment_correct / containment_total) if containment_total > 0 else "N/A"
    
    # MAE
    if not temporal_supported:
        metrics['Time_MAE'] = "N/A (non-temporal)"
    else:
        metrics['Time_MAE'] = np.mean(mae_diffs) if mae_diffs else 0.0
        
    return metrics

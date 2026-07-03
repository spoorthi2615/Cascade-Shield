import requests
import torch
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(r"d:\projects\cascade sheild")
from models.evaluate import evaluate
from models.cascade_net import CascadeNet

def test_dashboard_consistency(scenario_id=217):
    print(f"--- Testing Scenario {scenario_id} ---")
    
    # 1. Dashboard Output
    resp = requests.get(f'http://localhost:8000/api/dashboard/predict/{scenario_id}').json()
    gnn_probs = np.array(resp['gnn_probs'])
    seir_probs = np.array(resp['seir_probs'])
    gt = np.array(resp['ground_truth'])
    
    pos_mask = (gt == 1)
    neg_mask = (gt == 0)
    
    dash_gnn_auc = roc_auc_score(gt, gnn_probs)
    
    # Calculate custom P@K
    K = int(np.sum(gt))
    top_k_indices = np.argsort(gnn_probs)[-K:]
    true_positives = np.sum((gt == 1)[top_k_indices])
    dash_gnn_pk = true_positives / K if K > 0 else 0
    
    dash_seir_auc = roc_auc_score(gt, seir_probs)
    
    top_k_indices_seir = np.argsort(seir_probs)[-K:]
    true_positives_seir = np.sum((gt == 1)[top_k_indices_seir])
    dash_seir_pk = true_positives_seir / K if K > 0 else 0
    
    print(f"Dashboard GNN  | AUC: {dash_gnn_auc:.4f} | P@K: {dash_gnn_pk:.4f}")
    print(f"Dashboard SEIR | AUC: {dash_seir_auc:.4f} | P@K: {dash_seir_pk:.4f}")

if __name__ == '__main__':
    test_dashboard_consistency()

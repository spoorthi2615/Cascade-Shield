import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss to handle the extreme class imbalance (91.8% negative / safe nodes).
    Uses standard defaults from Lin et al.:
    - gamma=2.0: Aggressively down-weights the gradient contribution of easily classified negative nodes,
      forcing the network to focus on the rare, hard-to-predict cascade paths.
    - alpha=0.25: Provides a baseline weighting for the minority positive class.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, weights=None):
        # inputs are expected to be raw logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class
        
        # alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if weights is not None:
            focal_loss = focal_loss * weights
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def masked_mse_loss(pred_time, true_time, target_labels):
    """
    Computes MSE loss for the regression head strictly on nodes that actually
    got infected (where target_labels == 1.0).
    Never-infected nodes are masked out to avoid diluting the gradient with meaningless targets.
    """
    mask = target_labels == 1.0
    
    # If the cascade is size 1 (origin only) or 0, masking might leave extremely few nodes.
    # We must ensure we don't return NaN if mask sum is 0.
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_time.device, requires_grad=True)
        
    masked_pred = pred_time[mask]
    masked_true = true_time[mask]
    
    return F.mse_loss(masked_pred, masked_true)

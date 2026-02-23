import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence

from mmseg.registry import MODELS
from mmseg.models.losses.utils import weight_reduce_loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
# if you have a helper to get class weights, import it; otherwise class_weight can be list/tensor
# from .utils import get_class_weight
import torch
import torch.nn.functional as F
from mmseg.models.losses.utils import weight_reduce_loss

def pem_binary_loss(pred,
                    label,
                    weight=None,
                    class_weight=None,
                    reduction='mean',
                    avg_factor=None,
                    ignore_index=-100,
                    avg_non_ignore=False):
    """
    PyTorch-style BCEWithLogits one-way loss with ignore_index
    (faithfully matches how cross_entropy handles ignore_index).
    """

    assert pred.dim() == 4 and label.dim() == 3, \
        f"Expected pred [N,C,H,W], label [N,H,W], got {pred.shape}, {label.shape}"
    N, C, H, W = pred.shape

    # ensure label long dtype
    label = label.long()

    # Flatten for easy indexing
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # [N*H*W, C]
    label_flat = label.view(-1)  # [N*H*W]

    # valid indices (non-ignored)
    valid_mask = label_flat != ignore_index
    if not valid_mask.any():
        return pred.new_tensor(0.0)

    pred_valid = pred_flat[valid_mask]      # [num_valid, C]
    label_valid = label_flat[valid_mask]    # [num_valid]

    # gather logits of the correct class for valid pixels
    pred_selected = pred_valid.gather(1, label_valid.unsqueeze(1)).squeeze(1)

    # targets for BCEWithLogits: same as CE, 1 for correct class
    target_valid = torch.zeros_like(pred_selected, dtype=pred_selected.dtype, device=pred_selected.device)
    # compute BCE loss per-pixel (no reduction)
    loss = F.binary_cross_entropy_with_logits(pred_selected, target_valid, reduction='none')
    # optional pixel weighting
    if weight is not None:
        weight_flat = weight.view(-1)
        weight_valid = weight_flat[valid_mask].float()
    else:
        weight_valid = None

    # class weighting (optional)
    if class_weight is not None:
        if not isinstance(class_weight, torch.Tensor):
            class_weight = torch.tensor(class_weight, device=pred.device, dtype=pred.dtype)
        label_weights = class_weight[label_valid]
        loss = loss * label_weights

    # average factor: same as PyTorch CE (divide by num_valid if mean)
    if (avg_factor is None) and reduction == 'mean':
        if class_weight is None:
            if avg_non_ignore:
                avg_factor = valid_mask.sum().item()
            else:
                avg_factor = label.numel()

        else:
            # the average factor should take the class weights into account
            label_weights = torch.stack([class_weight[cls] for cls in label
                                         ]).to(device=class_weight.device)

            if avg_non_ignore:
                label_weights[label == ignore_index] = 0
            avg_factor = label_weights.sum()
    
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight_valid, reduction=reduction, avg_factor=avg_factor)

    return loss




@MODELS.register_module()
class OneWayLogisticLoss(nn.Module):
    """
    Compute one-way logisitic loss for one or several prior estimator modules

    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_one_way_logistic',
                 avg_non_ignore=False):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        # keep a python list or tensor of class weights if provided
        self.class_weight = class_weight
        self.avg_non_ignore = avg_non_ignore
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """
        Args:
            preds: pem_estimates is a list or None
            label: [N, H, W] integer labels
            weight: optional per-pixel weight map (N,H,W)
            avg_factor: explicit avg factor passed to pem loss
            reduction_override: override reduction
            ignore_index: ignored label value
        Returns:
            tensor: one-way logisitic loss 
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        
        # pem losses
        if cls_score:
            # sum over pem modules (each returns a scalar)
            pem_losses = []
            for pem_estimate in cls_score:
                pem_l = pem_binary_loss(
                                        pem_estimate,
                                        label,
                                        weight,
                                        class_weight=class_weight,
                                        reduction=reduction,
                                        avg_factor=avg_factor,
                                        avg_non_ignore=self.avg_non_ignore,
                                        ignore_index=ignore_index,
                                        **kwargs)
                pem_losses.append(pem_l)
            pem_loss_total = sum(pem_losses) / len(pem_losses)
        else:
            pem_loss_total = main_pred.new_tensor(0.0)
        return self.loss_weight * pem_loss_total
    @property
    def loss_name(self):
        return self._loss_name

import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple
import inspect

import torch
import torch.nn as nn
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from mmseg.models.losses.accuracy import accuracy
from mmseg.models.utils import resize

COMPATIBLE_HEADS = [
                    'FCNHead',
                    'UPerHead',
                    'ASPPHead',
                    'DepthwiseSeparableASPPHead',
                    'PSPHead',
                    'DNLHead',
                    'NLHead',
                    'APCHead',
                    'ANNHead',
                    'SETRUPHead',
                    'EMAHead',
                ]

@MODELS.register_module()
class DummyLoss(nn.Module):
    """A dummy zero loss that plays nicely with mmseg's loss parser."""
    def __init__(self, loss_name='_loss_dummy', loss_weight=0.0, **kwargs):
        super().__init__()
        self._loss_name = loss_name
        self.loss_weight = loss_weight

    def forward(self, preds, batch_data_samples=None, **kwargs):
        zero_loss = torch.tensor(
            0.0, device=preds.device, dtype=preds.dtype, requires_grad=False
        )
        return zero_loss

    @property
    def loss_name(self):
        return self._loss_name

@MODELS.register_module()
class MultiHeadWrapper(BaseDecodeHead):
    """Wrapper that attaches multiple prior estimator modules to an existing
    MMSeg decode head.

    Training:
        Returns (main_prediction, NPE estimates)

    Inference:
        Normalizes and merges NPE estimates into the final prediction.
    """

    def __init__(self,  
                base_head,
                num_pem=0,
                pem_weight=None,
                pem_type='auto',
                scale_factor=1,
                **kwargs):
        """
        Args:
            base_head (dict): Config for the primary decode head.
            num_pem (int): Number of prior estimator heads to attach.
            pem_weight (float or None): Loss weight for one-way logistic.
            pem_type (str): PEM type. Must be either:- "auto" - "FCNHead"
            scale_factorr (float): Scaling used when merging NPE outputs at inference.
        """
        base_type = base_head['type']
        if base_type not in COMPATIBLE_HEADS:
            raise TypeError(
                            f"MultiHeadWrapper is not compatible with decode head '{base_type}'."
                            " Please use a compatible head or modify the wrapper."
                        )
        valid_args = inspect.signature(BaseDecodeHead.__init__).parameters.keys()
        base_kwargs = {k: v for k, v in base_head.items() if k in valid_args}
        if base_type in ['UPerHead', 'ANNHead']:
            base_kwargs['input_transform'] = 'multiple_select'
        if base_type in ['DNLHead', 'NLHead']:
            base_kwargs['num_convs'] = 2
        super().__init__(**base_kwargs)
        
        self.scale_factor = scale_factor
        self.num_pem = num_pem

        # -----------------------
        # Build main + PEMs
        # -----------------------
        
        base_head_copy = base_head.copy()
        base_head_copy['loss_decode'] = dict(type='DummyLoss')  # satisfy BaseDecodeHead
        self.main_head = MODELS.build(base_head_copy)
        
        base_head_copy['norm_cfg'] = None
        
        if pem_type =='FCNHead':
            base_head_copy = self.prepare_pem_module(base_head_copy)
        
        
        self.pems = nn.ModuleList([
            MODELS.build(base_head_copy) for _ in range(num_pem)
        ])
        # -----------------------
        # Wrapper-level loss
        # -----------------------
        self.loss_decode = nn.ModuleList()

        if isinstance(base_head['loss_decode'], dict):
            
            if base_head['loss_decode'].get('type') != 'CrossEntropyLoss':
                raise TypeError('expected the base head loss to be CrossEntropyLoss')
            else:
                self.loss_decode.append(MODELS.build(base_head['loss_decode']))
                loss_docode_update = base_head['loss_decode']
        
        elif isinstance(base_head['loss_decode'], (list, tuple)):
            list_of_losses = list()
            for loss in base_head['loss_decode']:
                self.loss_decode.append(MODELS.build(loss))
                if loss.get('type') == 'CrossEntropyLoss':
                    loss_docode_update = loss
                list_of_losses.append(loss.get('type'))
            if not any(i == 'CrossEntropyLoss' for i in list_of_losses):
                raise TypeError('expected at least one of the base head loss to be CrossEntropyLoss')

        
        loss_docode_update['type'] = 'OneWayLogisticLoss'
        loss_docode_update['loss_name'] = 'loss_one_way_logistic'
        
        # if pem_weight is not provided, its loss is similar to ce loss weight
        if pem_weight:
            loss_docode_update['loss_weight'] = pem_weight
        
        
        if num_pem > 0:
            self.loss_decode.append(MODELS.build(loss_docode_update))
        
        
        # -----------------------
        # Initialize weights
        # -----------------------
        self.init_weights()

    # -----------------------
    # Initialization
    # -----------------------
    def init_weights(self):
    
        def _init_fn(m):
        
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, std = 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.apply(_init_fn)

    def prepare_pem_module(self, base_head):
        in_channels = base_head['in_channels']
        if isinstance(in_channels, (list, tuple)):
            in_channels = in_channels[-1]
        in_index = base_head['in_index'] 
        if isinstance(in_index, (list, tuple)):
            in_index = in_index[-1] 
               
        pem_module=dict(
        type='FCNHead',
        in_channels=in_channels,
        in_index=in_index,
        channels=base_head['channels'],
        num_convs=1,
        concat_input=False,
        dropout_ratio=base_head['dropout_ratio'],
        num_classes=base_head['num_classes'],
        norm_cfg=None,
        align_corners=base_head['align_corners'],
        loss_decode=dict(
            type='DummyLoss'))
            
        return pem_module

                
    # -----------------------
    # Forward
    # -----------------------
    def forward(self, inputs):
        main_pred = self.main_head(inputs)

        if self.training:
            pem_estimates = [h(inputs) for h in self.pems]
            
            return main_pred, pem_estimates
        else:

            if self.num_pem:
                dimentions = tuple(range(1,main_pred.ndim))
                pem_estimate_norm = []
                for h in self.pems:
                    
                    pem_estimate = h(inputs)
                    pem_estimate = resize(pem_estimate, size=main_pred.shape[2:], mode='bilinear', align_corners=self.align_corners)
                    pem_estimate_mean = pem_estimate - pem_estimate.mean(dim = dimentions, keepdim = True)
                    pem_estimate_norm.append(pem_estimate_mean / self.scale_factor)
                main_pred = main_pred + sum(pem_estimate_norm) / self.num_pem
                
            return main_pred

    # -----------------------
    # Loss computation
    # -----------------------
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Compute segmentation loss and metrics."""
        seg_logits, pem_estimatess = self.forward(inputs)  # returns (main_pred, pem_estimates)
        return self.loss_by_feat((seg_logits, pem_estimatess ), batch_data_samples)

    def loss_by_feat(self, seg_logits: Tuple[Tensor, list[Tensor]],
                     batch_data_samples: SampleList) -> dict:
        main_pred, pem_estimates = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        target_size = seg_label.shape[2:]
        main_pred = resize(main_pred, size=target_size,
                           mode='bilinear', align_corners=self.align_corners)
        pem_estimates = [
            resize(adj, size=target_size,
                   mode='bilinear', align_corners=self.align_corners)
            for adj in pem_estimates
        ]

        if self.sampler is not None:
            seg_weight = self.sampler.sample(main_pred, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)


        for loss_decode in self.loss_decode:

            if loss_decode.loss_name == 'loss_one_way_logistic':
                loss[loss_decode.loss_name] = loss_decode(
                    pem_estimates,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        main_pred,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        main_pred,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
        
        with torch.no_grad():
            acc = 0
            B = main_pred.shape[0]

            for b in range(B):
                merged = main_pred[b]
                if self.num_pem > 0:
                    for adj in pem_estimates:
                        merged = merged + adj[b] / (self.num_pem * self.scale_factor)

                # accuracy expects [N, C, H, W] for pred, [N, H, W] for target
                acc += accuracy(merged.unsqueeze(0), seg_label[b].unsqueeze(0),
                               ignore_index=self.ignore_index)
            

            loss['acc_seg'] = acc  / B 
        
        return loss

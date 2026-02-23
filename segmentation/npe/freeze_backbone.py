from mmengine.hooks import Hook
import torch.nn as nn
import torch  
        
from mmengine.registry import HOOKS, MODELS
from mmengine.runner import load_checkpoint
from collections import OrderedDict

@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    """Completely freeze backbone parameters and normalization layers.

    This hook:
    - Sets requires_grad = False for all backbone parameters
    - Puts backbone and its BatchNorm/SyncBN layers in eval() mode
    - Disables running statistics updates (track_running_stats = False)
    - Logs how many parameters were frozen
    """

    def before_train(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module  # unwrap DDP if needed

        if not hasattr(model, 'backbone'):
            runner.logger.warning('⚠️ Model has no backbone attribute — skipping freeze.')
            return

        backbone = model.backbone
        frozen_params = 0

        # 1️⃣ Freeze all backbone params
        for p in backbone.parameters():
            p.requires_grad = False
            frozen_params += p.numel()

        # 2️⃣ Disable BN running stats and set eval mode
        for m in backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
                m.track_running_stats = False
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad = False
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad = False
        # 3️⃣ Put backbone fully in eval mode
        backbone.eval()
        runner.logger.info(
            f'✅ Backbone frozen successfully: '
            f'{frozen_params/1e6:.2f}M parameters | '
            f'BN layers static (no running mean/var updates).'
        )


@HOOKS.register_module()
class PretrainedRemapHook(Hook):
    """Load pretrained decode_head weights into MultiHeadWrapper.main_head."""

    def __init__(self, checkpoint, freeze_after_load=True):
        self.checkpoint = checkpoint
        self.freeze_after_load = freeze_after_load

    def before_train(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module  # handle DDP wrapping

        ckpt = torch.load(self.checkpoint, map_location='cpu', weights_only = False)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        new_state_dict = OrderedDict()
        loaded = 0

        # Remap decode_head.* → decode_head.main_head.*
        for k, v in ckpt.items():
            if k.startswith('decode_head.'):
                new_k = k.replace('decode_head.', 'decode_head.main_head.')
                new_state_dict[new_k] = v
                loaded += 1
            else:
                new_state_dict[k] = v

        msg = model.load_state_dict(new_state_dict, strict=False)
        runner.logger.info(
            f"✅ PretrainedRemapHook loaded {loaded} decode_head params from {self.checkpoint}"
        )
        runner.logger.info(f"Ignored / missing keys:\n{msg}")

        # --- Optionally freeze ---
        if self.freeze_after_load:
            head = model.decode_head.main_head
            for p in head.parameters():
                p.requires_grad = False
            head.eval()
            runner.logger.info("decode_head.main_head parameters have been frozen and set to eval mode.")


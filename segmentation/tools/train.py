import os
import sys
import os.path as osp
import argparse

llr_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, llr_root)

from npe import *

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmseg.registry import RUNNERS

# parse args

def parse_args():
    parser = argparse.ArgumentParser(description='Custom MMseg Trainer')

    parser.add_argument('base_config', type=str,
                        help='Path to a mmseg base config (U-Net, etc.)')

    parser.add_argument('--load-from', type=str, default=None,
                        help='Checkpoint for backbone/decoder initialization.')

    parser.add_argument('--work-dir', type=str, default=None,
                        help='Where to save logs and models')

    parser.add_argument('--freeze-backbone', action='store_true',
                        help = 'freeze backbone')
    parser.add_argument('--freeze-decode', action='store_true',
                        help = 'freeze main decode head')
    parser.add_argument('--num-pem', type=int, default=1,
                        help = 'number of prior estimator heads, default is 1')
    parser.add_argument('--pem-type', type=str, default='auto',
                        help = 'type of prior estimator heads. Default is auto. Options are "auto" and "FCNHead"')
    parser.add_argument('--scale-factor', type=int, default=1,
                        help = 'Scale factor "s"')

    parser.add_argument('--cfg-options',
                        nargs='+', action=DictAction,
                        help='Additional overrides (same as mmseg).')

    return parser.parse_args()

def update_cfg(args):
    
    # Load base config
    cfg = Config.fromfile(args.base_config)
    # Custom imports for your module
    cfg.custom_imports = dict(
        imports=[
            'npe.multi_head_wrapper',
            'npe.multi_head_custom_loss',
            'npe.freeze_backbone',
        ],
        allow_failed_imports=False,
    )


    # model modifications
    cfg.model.auxiliary_head = None

    cfg.model.decode_head = dict(
        type='MultiHeadWrapper',
        base_head=cfg._cfg_dict['model']['decode_head'],  # inherit original
        num_pem = args.num_pem,
        pem_type=args.pem_type,
        pem_weight=None,
        scale_factor=args.scale_factor,
    )

    
    # optimizer adjustments
    backbone_lr = 0.0 if args.freeze_backbone else cfg.optim_wrapper.optimizer.lr
    base_decode_lr = 0.0 if args.freeze_decode else cfg.optim_wrapper.optimizer.lr
    cfg.optim_wrapper.paramwise_cfg = dict(
        custom_keys={
        'backbone': dict(lr_mult=backbone_lr), 
        'decode_head.base_head': dict(lr_mult=base_decode_lr)
        }
    )
    # Change scheduler for STARE and ADE20k datasets
    dataset_name = None
    try:
        dataset_name = cfg.dataset_type.lower()
    except:
        pass

    interval = None
    max_iters = None
    milestones = None

    if dataset_name == 'staredataset':
        interval = 200
        max_iters = 600
        milestones = [200, 400]        

    elif dataset_name == 'ade20kdataset':
        interval = 1250
        max_iters = 2500
        milestones = [1250]

    if interval is not None:
        cfg.train_cfg.max_iters = max_iters
        cfg.train_cfg.val_interval = interval

        cfg.param_scheduler = [
            dict(
                type='MultiStepLR',
                by_epoch=False,
                gamma=0.1,
                milestones=milestones,
            )
        ]

    # hooks for freezing backbone and main decode head
    custom_hooks = []
    if args.freeze_backbone:
        custom_hooks.append(dict(type='FreezeBackboneHook', priority=50))

    if args.load_from:
        custom_hooks.append(dict(
            type='PretrainedRemapHook',
            checkpoint=args.load_from,
            freeze_after_load=args.freeze_decode,
            priority=40
        ))

    if len(custom_hooks) > 0:
        cfg.custom_hooks = custom_hooks

    # Data loading override
    cfg.train_dataloader.batch_size = 16

    # work_dir
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.base_config))[0])

    # load_from injection
    if args.load_from:
        cfg.load_from = args.load_from

    # allow overrides
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    base_name = osp.splitext(osp.basename(args.base_config))[0]
    #directory = 'llr_configs'
    #os.makedirs(directory, exist_ok=True)

    file_name = f'{base_name}_npe_{args.num_pem}.py'
    #config_path = osp.join(directory, file_name)

    cfg.dump(file_name)
    updated_config = Config.fromfile(file_name)
    os.remove(file_name)
    return updated_config
    

def main():
    args = parse_args()

    cfg = update_cfg(args)
    # build runner
    if 'runner_type' in cfg:
        runner = RUNNERS.build(cfg)
    else:
        runner = Runner.from_cfg(cfg)

    runner.train()


if __name__ == '__main__':
    main()

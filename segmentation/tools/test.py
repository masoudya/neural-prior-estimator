import os
import sys
import os.path as osp
import argparse

llr_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, llr_root)

from npe import *

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Custom MMSeg Test Runner')

    parser.add_argument(
        'train_work_dir',
        help='Path to the training work directory containing config + checkpoints'
    )

    parser.add_argument(
        '--scale-factor',
        type=float,
        default=1.0,
        help='scale factor override for MultiHeadWrapper'
    )

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none'
    )

    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()


    # Locate config and checkpoint

    work_dir = args.train_work_dir
    if not osp.isdir(work_dir):
        raise FileNotFoundError(f'Work directory not found: {work_dir}')

    # find the config (.py)
    config_file = None
    for f in os.listdir(work_dir):
        if f.endswith('.py'):
            config_file = osp.join(work_dir, f)
            break
    if config_file is None:
        raise FileNotFoundError('No config .py file found in work directory.')

    # read last checkpoint path
    last_ckpt_file = osp.join(work_dir, 'last_checkpoint')
    if not osp.isfile(last_ckpt_file):
        raise FileNotFoundError('last_checkpoint file not found.')

    with open(last_ckpt_file, 'r') as f:
        ckpt_path = f.read().strip()
    if not osp.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    #Load config
    cfg = Config.fromfile(config_file)
    cfg.load_from = ckpt_path
    cfg.work_dir = work_dir
    cfg.launcher = args.launcher


    # Optional TTA

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model


    # Override MultiHeadWrapper normalizer
    # Only if the decode_head is a MultiHeadWrapper
    model_head = cfg.model.get('decode_head', {})
    if isinstance(model_head, dict) and model_head.get('type') == 'MultiHeadWrapper':
        model_head['scale_factor'] = args.scale_factor


    # Run test
    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()

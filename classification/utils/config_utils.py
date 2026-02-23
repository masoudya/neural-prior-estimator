

import os
import yaml
import re
import argparse

class Config:
    """A wrapper for dict/attribute-style access."""
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def update_if_provided(self, **kwargs):
        """Update only parameters that are not None."""
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)
                print(k, v)

    def merge_from(self, other):
        """Merge parameters from another Config object."""
        for k, v in other.__dict__.items():
            setattr(self, k, v)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return yaml.dump(self.__dict__, sort_keys=False)

# Regex to detect floats and scientific notation
_float_pattern = re.compile(r'^[-+]?\d*\.?\d+(e[-+]?\d+)?$', re.IGNORECASE)

def _auto_cast(value):
    """Convert strings into int/float/bool when possible."""
    if isinstance(value, str):
        val = value.strip()

        # Booleans
        if val.lower() in ['true', 'false']:
            return val.lower() == 'true'

        # Ints
        if val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
            return int(val)

        # Floats or scientific notation
        if _float_pattern.match(val):
            try:
                return float(val)
            except ValueError:
                return val  # fallback to string if conversion fails

        # Leave others as string
        return val

    elif isinstance(value, list):
        return [_auto_cast(v) for v in value]

    elif isinstance(value, dict):
        return {k: _auto_cast(v) for k, v in value.items()}

    return value

def load_yaml_config(path):
    """Load YAML config safely and auto-cast numeric strings."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file '{path}' is not a valid mapping.")

    casted = _auto_cast(data)
    return Config(**casted)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simplified OOP Trainer (DataParallel)")
    parser.add_argument('--dataset', type=str, default = None, help="Dataset")
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--imb-factor', type=int)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--num-pem', type=int, default = 0, help = 'number of prior estimator modules')
    parser.add_argument('--save-checkpoint',  action='store_true', help = 'Whether to save the model or not')
    parser.add_argument('--loss-function', default = 'CE', type=str, help='CE, LA, etc.')
    parser.add_argument('--cfg-options', nargs=argparse.REMAINDER)
    return parser.parse_args()

def parse_inline_opts(opts_list):
    """
    Parse inline overrides like ['lr=0.01', 'dataset=cifar100'] into a dict.
    """
    opts_dict = {}
    for item in opts_list:
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Auto-cast: bool, int, float, or string
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif re.match(r'^-?\d+$', value):
            value = int(value)
        elif re.match(r'^-?\d*\.?\d+(e[-+]?\d+)?$', value, re.IGNORECASE):
            value = float(value)

        opts_dict[key] = value
    return opts_dict

def get_config(path = None):
    """
    Load configs
    """
    
    args = parse_args()
    # --- base configs for distributed training are loaded
    base_config_path = os.path.join('config', 'base', 'dist_train.yaml')
    
    config = load_yaml_config(base_config_path)

    # --- load config from path (if provided) ---
    if path:
        config_path = path

    # --- if path not provided, load from args
    elif args.dataset:
        config_path = os.path.join('config', f'{args.dataset}', f'{args.dataset}.yaml')
    
    # --- else, load default cifar100 configs  
    else:
        config_path = os.path.join('config', 'cifar100', 'cifar100.yaml')
    
    other_config = load_yaml_config(config_path)
    config.merge_from(other_config)
    # --- Override with CLI args ---
   
    config.update_if_provided(
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        imb_factor=args.imb_factor,
        weight_decay=args.weight_decay,
        num_pem=args.num_pem,
        loss_function=args.loss_function,
        save_checkpoint = args.save_checkpoint
    )
        
    # --- Apply inline key=value overrides ---
    if args.cfg_options:
        inline_overrides = parse_inline_opts(args.cfg_options)
        config.update_if_provided(**inline_overrides)
    
    if config.num_pem < 0:
        raise ValueError(f"numer of pem must be a positive value")
    
    return config, args

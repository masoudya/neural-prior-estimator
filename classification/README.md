# Long-Tailed Classification with Neural Prior Estimator (NPE)


This directory contains the classification instantiation of **Neural Prior Estimator (NPE)** for long-tailed recognition.


The implementation integrates learned feature-conditioned priors into prediction via adaptive logit adjustment, enabling bias-aware learning under class imbalance.

## Installation

1. Clone the repository and change directory:
   ```bash
   git clone https://github.com/masoudya/neural-prior-estimator.git
   cd neural-prior-estimator/classification
   ```
2. Install dependencies from the repository root:
   ```bash
   pip install -r ../requirements.txt
   ```
## Quick Start

To traine on cifar-10 using one Prior Estimation Module (PEM) with default settings:

```bash
python train.py --dataset cifar10 --num-pem 1
```
the result can be visualized by
```bash
python report.py logs/result/<experiment_name>.npy --visualize-train --visualize-test
```

## Configuration System

Training behavior is fully controlled through YAML configuration files and optional command-line overrides.  
The system merges multiple sources of configuration in a structured order to ensure reproducibility and flexibility.


### Configuration Loading Order

At runtime, parameters are resolved in the following order:

1. **Base configuration**:
`config/base/dist_train.yaml` controls device configuration.
2. **Dataset-specific configuration**:
`config/<dataset>/<dataset>.yaml`
which defines dataset-specific settings and training hyperparameters. <dataset> is automatically selected when `--dataset` is provided.

3. **Command-line arguments**:
Override individual parameters such as learning rate or number of PEM modules.

4. **Inline overrides**:
Arbitrary key-value pairs passed via `--cfg-options`.

Later sources override earlier ones.

### YAML Configuration Structure

Each dataset configuration file defines dataset settings, model parameters, and training hyperparameters.

### Command-Line Overrides

Common training parameters can be overridden without modifying YAML files.

Available override arguments are:

| Argument | Description |
|---|---|
| `--dataset` | Dataset name |
| `--lr` | Learning rate |
| `--batch-size` | Batch size |
| `--num-epochs` | Number of training epochs |
| `--imb-factor` | Imbalance factor |
| `--weight-decay` | Weight decay |
| `--num-pem` | Number of Prior Estimation Modules |
| `--loss-function` | Loss type (CE or LA) |
| `--save-checkpoint` | Save trained model |

Example:
```bash
python train.py \
    --dataset cifar10 \
    --lr 0.05 \
    --batch-size 256 \
    --num-epochs 300 \
    --num-pem 2 \
    --save-checkpoint
```

For advanced usage, arbitrary parameters can be overridden using key-value syntax.

Example:
```bash
python train.py \
    --dataset cifar10 \
    --cfg-options lr=0.01 batch_size=64 momentum=0.95
```

### Configuration Object
All configuration values are stored in a unified configuration object with attribute-style access. For example:

+ config.lr
+ config.batch_size
+ config.num_pem

## Datasets

The framework is compatible with [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html), [cifar100](https://www.cs.toronto.edu/~kriz/cifar.html), [iNaturalist 2018](https://github.com/visipedia/inat_comp/blob/master/2018/README.md),
[Places](http://places.csail.mit.edu/) and [ImageNet](https://www.image-net.org/) datasets.

By default, datasets are expected at: `classification/data/`

The dataset path is controlled by the configuration parameter:

```yaml

data_path: ./data/<dataset_name>
```

### Expected Directory Structure

Example for CIFAR datasets:

```text
classification/
└── data/
    ├── cifar10/
    └── cifar100/
```

The directory name must match the value specified in the configuration file.

### Changing Dataset Location

To use a different dataset path, modify the configuration file: `data_path: /path/to/your/dataset`

Or override from command line:
```bash
python train.py --dataset cifar10 --cfg-options data_path=/path/to/data
```

## Logs & Reporting

During training, NPE saves experiment logs as a NumPy file in the default directory:
`logs/result/`

Each run is assigned a unique timestamp:
`logs/result/run_<YYYY-MM-DD_HH-MM-SS>.npy`

### Log File Contents

The `.npy` file stores a dictionary with the following keys:

| Key | Description |
|-----|------------|
| `timestamp` | Run timestamp for reproducibility |
| `backbone_path` | Path to backbone checkpoint (if any) |
| `block_path` | Path to additional model blocks (if any) |
| `train_losses` | Array of training loss per epoch |
| `pem_losses` | Array of PEM module loss per epoch |
| `val_losses` | Array of validation loss per epoch |
| `val_accs` | Array of validation accuracy per epoch |
| `per_class_accs` | Array of per-class accuracies for each epoch `[num_epochs, num_classes]` |
| `config` | Dictionary of all config parameters used in the run |

The logging format is designed to support long-tailed evaluation metrics used in the NPE paper, including per-class and group-wise accuracy analysis.

### Report Generation

Use `report.py` to summarize results and optionally visualize training and validation curves.

```bash
python report.py <path_to_log.npy> [--visualize-train] [--visualize-test]
```
#### Arguments

- `<path_to_log.npy>`:	Path to the .npy log file generated during training
- `--visualize-train`:	Plot training loss curve
- `--visualize-test`:	Plot validation accuracy and final group accuracies

Example:
```bash
python report.py logs/result/run_2026-02-23_12-00-00.npy --visualize-train --visualize-test
```

The log tracks per-class accuracy. For long-tailed analysis, classes are grouped as:

   - Head classes → head_class_idx from config
   - Medium classes → med_class_idx from config
   - Tail classes → tail_class_idx from config

Group accuracy is computed as the average of per-class accuracies in each group at the final epoch.
Visualization

### Visualization

Available plots include:

- Training loss curve
- Validation accuracy curve
- Final group accuracy comparison (Head / Medium / Tail)

These visualizations help assess convergence and long-tailed performance behavior.



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

for training cifar10 dataset, with one _Prior Estimator Module (PEM)_, and default configuration, run:

```bash
python train.py --dataset cifar10 --num-pem 1
```
the result can be visualized by
```bash
python report.py logs/result/<expriment_name>.npy --visualize-train
```
## Datasets
Supported datasets are [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html), [cifar100](https://www.cs.toronto.edu/~kriz/cifar.html), [iNaturalist 2018](https://github.com/visipedia/inat_comp/blob/master/2018/README.md),
[Places](http://places.csail.mit.edu/) and [ImageNet](https://www.image-net.org/).

## Configuration System

Training behavior is fully controlled through YAML configuration files and optional command-line overrides.  
The system merges multiple sources of configuration in a structured order to ensure reproducibility and flexibility.


### Configuration Loading Order

At runtime, parameters are resolved in the following order:

1. **Base configuration**:
`config/base/dist_train.yaml` controls device configuration.
2. **Dataset-specific configuration**:
`config/<dataset>/<dataset>.yaml`
which controls mainly hyperparameters. <dataset> is automatically selected when `--dataset` is provided.

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
| Argument |	Description |
| :---         |     :---      |
| --dataset |	Dataset name    |
| --lr |	Learning rate     |
| --batch-size |	Batch size  |
|--num-epochs |	Number of training epochs     |
| --imb-factor |	Imbalance factor   |
| --weight-decay	| Weight decay      |
| --num-pem |	Number of Prior Estimation Modules  |
| --loss-function	| Loss type ( CE or LA)     |
| --save-checkpoint |	Save trained model    |

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

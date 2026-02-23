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


## Configuration
All configuration files are in `classification/config` folder. 
- For device configuration, change `classification/config/base/dist_train.yaml` file.
- For hyperparameters, change `classification/config/<dataset>/<dataset>.yaml`


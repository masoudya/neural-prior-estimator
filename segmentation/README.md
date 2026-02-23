# Semantic Segmentation with Neural Prior Estimator (NPE)

This directory contains the semantic segmentation implementation of **Neural Prior Estimator (NPE)** built on top of MMSegmentation.

The method integrates learned feature-conditioned priors into dense prediction, enabling imbalance-aware semantic segmentation.


## Installation

This implementation depends on MMSegmentation.

**Step 1:** Install MMSegmentation

Follow the official mmsegmentation [installation guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation). 
Make sure MMSegmentation runs correctly before proceeding.

**Step 2:** Clone Neural Prior Estimator repository

```bash
git clone https://github.com/masoudya/neural-prior-estimator.git
cd neural-prior-estimator/segmentation
```
### Repository Layout Requirement
The expected directory layout is:
```text
workspace/
├── mmsegmentation/
│ ├── mmseg/
│ ├── tools/
│ └── ...
│
└── neural-prior-estimator/
    ├── classification/
    ├── segmentation/
    └── ...
```

## Dataset Preparation

Datasets must be placed inside: `segmentation/data/`

Dataset preparation follows the same structure and conventions as MMSegmentation. 
Refer to the official [dataset preparation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets)
guide.
```text
segmentation/
│
├── data/ # dataset directory (MMSegmentation format)
├── npe/ # NPE modules and wrappers
├── tools/ # training and evaluation scripts
│ ├── train.py
│ └── test.py
└── README.md
```

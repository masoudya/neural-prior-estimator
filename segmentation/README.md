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
This layout is required because the segmentation pipeline imports MMSegmentation modules.
## Dataset Preparation
Dataset format and structure follow,
[MMSegmentation conventions](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets)
but the storage location is `neural-prior-estimator/segmentation/data/`. ⚠️ DO NOT place dataset in `mmsegmentation/data`. 

```text
segmentation/
│
├── data/ 
├── npe/ 
├── tools/
│ ├── train.py
│ └── test.py
└── README.md
```


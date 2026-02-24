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
    │ ├── data/ 
    │ ├── npe/ 
    │ ├── tools/
    │ │ ├── train.py
    │ │ └── test.py
    └── README.md
    └── ...
```
This layout is required because the segmentation pipeline imports MMSegmentation modules.
## Dataset Preparation
Dataset format and structure follow,
[MMSegmentation conventions](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets)
but the storage location is `neural-prior-estimator/segmentation/data/`. ⚠️ DO NOT place dataset in `mmsegmentation/data`. 

## Training

Training is performed using a standard MMSegmentation base configuration with NPE modifications applied at runtime.

### Basic Usage
```bash
python tools/train.py <base_config.py> [OPTIONS]
```
Example:
```bash
python tools/train.py \
    configs/unet/unet_r50.py \
    --num-pem 2 \
    --scale-factor 2 \
    --work-dir work_dirs/unet_npe
```
### How NPE Integrates with MMSegmentation

This implementation does not replace MMSegmentation models.
Instead, it **wraps the existing decode head** with a Neural Prior Estimator module at runtime.

The training script performs the following steps:

  1. Loads a standard MMSegmentation base configuration.
   2. Removes the original auxiliary head (if present).
   3. Wraps the main decode head using MultiHeadWrapper.
   4. Attaches one or more Prior Estimator Modules (PEMs).
   5. Adjusts optimization and scheduler settings if required.

As a result, any segmentation model supported by can be used without modifying its config file.

#### Model Wrapping Mechanism

The original architecture:
```
Backbone → Decode Head → Segmentation Output
```
After applying NPE:
```
Backbone → MultiHeadWrapper
              ├── Original Decode Head
              └── Prior Estimator Module(s)
                     ↓
            Bias-adjusted segmentation logits
```
The Prior Estimator learns feature-conditioned priors that adjust class predictions to improve performance under imbalance.
Prior Estimator Module (PEM)

Each PEM receives the same feature representation as the main decode head and produces a learned prior that modifies prediction logits.

### Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `base_config` | str | — | Path to MMSegmentation base config (e.g., U-Net, DeepLabV3) |
| `--load-from` | str | None | Load pretrained checkpoint for initialization |
| `--work-dir` | str | Auto-generated | Directory for logs and checkpoints |
| `--freeze-backbone` | flag | False | Freeze backbone parameters during training |
| `--freeze-decode` | flag | False | Freeze original decode head (only NPE learns) |
| `--num-pem` | int | 1 | Number of Prior Estimator heads |
| `--pem-type` | str | auto | Prior estimator type: `auto` or `FCNHead` |
| `--scale-factor` | int | 1 | Scaling factor `s` for prior adjustment |
| `--cfg-options` | key=value pairs | None | Override config parameters (same format as MMSegmentation) |

#### PEM Type	Behavior
auto	Uses the same architecture as the base decode head
FCNHead	Uses a lightweight fully-connected segmentation head

Multiple PEMs can be attached simultaneously using --num-pem.
#### Freezing Behavior

The framework supports controlled training regimes:
Option	Effect
`--freeze-backbone`	Backbone parameters are not updated
`--freeze-decode`	Original decode head is frozen; only PEM learns
`none`	Full model training

### Automatic Hyperparameter Setting
If the dataset in the base config is ADE20K or STARE, training hyperparameters are automatically adjusted to match the settings used in the paper for reproducibility. No manual tuning is required.
When the dataset type in the base config matches the following:

STARE
```text
max_iters = 600
val_interval = 200
LR decay milestones: [200, 400]
```

ADE20K
```text
max_iters = 2500
val_interval = 1250
LR decay milestones: [1250]
```

These settings override scheduler and training iteration configuration automatically.

### Configuration Overrides

Similar to mmsegmentation, any parameter in the base config can be modified without editing files:
```bash
python tools/train.py <base_config.py> \
    --cfg-options optim_wrapper.optimizer.lr=0.0001
```
Multiple overrides are supported:
```bash
python tools/train.py <base_config.py> \
    --cfg-options \
    train_dataloader.batch_size=8 \
    model.decode_head.num_classes=19
```

---

### Output Directory

If `--work-dir` is not specified, outputs are saved to:`./work_dirs/<base_config_name>/`

This directory contains:
- training logs
- checkpoints
- final model weights

# Semantic Segmentation with Neural Prior Estimator (NPE)

This directory provides the semantic segmentation implementation of **Neural Prior Estimator (NPE)** built on top of 
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main).
NPE augments dense prediction models with learnable feature-conditioned class priors that adjust segmentation logits, improving performance under class imbalance.
The implementation does not modify base model definitions. Instead, it wraps existing decode heads at runtime and injects prior estimation modules.


## Installation

This implementation depends on MMSegmentation.

**Step 1:** Install MMSegmentation

Follow the official mmsegmentation [installation guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation). 
Make sure MMSegmentation runs correctly before proceeding.

**Step 2:** Clone Neural Prior Estimator

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

Training uses a standard MMSegmentation base configuration. NPE modules are attached automatically at runtime.

### Basic Usage
```bash
python tools/train.py <base_config.py> [OPTIONS]
```
Example:
```bash
python tools/train.py \
    ../../configs/unet/unet_r50.py \
    --num-pem 2 \
    --scale-factor 1 \
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

**Supported Base Heads**

At present, the wrapper is compatible with the following [decode heads](https://github.com/open-mmlab/mmsegmentation/tree/main/mmseg/models/decode_heads):

- `FCNHead`
- `UPerHead`
- `ASPPHead`
- `DepthwiseSeparableASPPHead`
- `PSPHead`
- `DNLHead`
- `NLHead`
- `APCHead`
- `ANNHead`
- `SETRUPHead`
- `EMAHead`

Attempting to wrap unsupported heads may lead to runtime errors.

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
Each PEM receives the same feature representation as the main decode head and produces a learned prior that modifies prediction logits.

### Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `base_config` | str | — | Path to MMSegmentation base config (e.g., U-Net, DeepLabV3) |
| `--load-from` | str | None | Load pretrained checkpoint for initialization |
| `--work-dir` | str | Auto-generated | Directory for logs and checkpoints |
| `--freeze-backbone` | flag | False | Freeze backbone parameters during training |
| `--freeze-decode` | flag | False | Freeze original decode head (only NPE learns) |
| `--num-pem` | int | 1 | Number of Prior Estimator Modules |
| `--pem-type` | str | auto | Prior estimator type: `auto` or `FCNHead` |
| `--scale-factor` | int | 1 | Scaling factor `s` for prior adjustment |
| `--cfg-options` | key=value pairs | None | Override config parameters (same format as MMSegmentation) |

#### PEM Type Behavior

| Value | Description |
|---|---|
| `auto` | Replicates architecture of the base decode head |
| `FCNHead` | Lightweight fully-connected segmentation head |

Multiple PEMs can be attached simultaneously using `--num-pem`.

### Automatic Hyperparameter Setting
To ensure reproducibility, training settings are automatically adjusted when specific datasets are detected in the base configuration.

- STARE:
```text
max_iters = 600
val_interval = 200
LR decay milestones: [200, 400]
```

- ADE20K:
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


## Evaluation

Model evaluation uses the trained work directory produced during training.  
The test runner automatically loads:

- the saved configuration
- the latest checkpoint
- model architecture with NPE modules
- final model weights
  
### Basic Usage

```bash
python tools/test.py <train_work_dir>
```
Example:
```bash
python tools/test.py work_dirs/unet_npe
```
### Command-Line Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `train_work_dir` | str | — | Path to training work directory containing config and checkpoints |
| `--scale-factor` | float | 1.0 | Override scaling factor of the NPE MultiHeadWrapper during evaluation |
| `--tta` | flag | False | Enable test-time augmentation |
| `--launcher` | str | none | Distributed launcher (`none`, `pytorch`, `slurm`, `mpi`) |
| `--local-rank` | int | 0 | Local process rank for distributed evaluation |

#### Test-Time Scaling

The `--scale-factor` argument overrides the scaling parameter of the NPE `MultiHeadWrapper` during evaluation.

- Larger values of `--scale-factor` **reduce the effect** of the effect of logit adjusment effect.
- This allows controlled analysis of adjustment strength without retraining.




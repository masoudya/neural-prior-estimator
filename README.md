# neural-prior-estimator
Official PyTorch implementation of Neural Prior Estimator (NPE) and NPE-LA for long-tailed classification and semantic segmentation.

Deep neural networks trained on imbalanced data implicitly encode skewed class priors. NPE estimates these priors directly from latent representations and integrates them into prediction through adaptive logit adjustment (NPE-LA). This provides a principled mechanism for improving performance on underrepresented classes without requiring empirical class counts or distribution-specific tuning.

## Paper
A preprint of the paper is available on [_Arxiv_](https://arxiv.org/abs/2602.17853).

# Experiments


## Long-tailed Classification

- Directory: [`classification/`](https://github.com/masoudya/neural-prior-estimator/classification)

- Instructions: See `classification/README.md` for installation, training, and evaluation.


## Imbalanced Semantic Segmentation

- Directory: `segmentation/`

- Instructions: See `segmentation/README.md` for installation, training, and evaluation

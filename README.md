<div align="center">

# Unveiling Transferability in Trajectory Prediction<br>via Latent Scene Embeddings

<p>
  <a href="https://arxiv.org/abs/2606.30777"><img src="https://img.shields.io/badge/arXiv-2606.30777-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://westny.github.io/transferatlas/"><img src="https://img.shields.io/badge/Project-Page-1f6feb?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project Page"></a>
</p>

<img src="https://img.shields.io/badge/ECCV-2026-2c8c4a?style=flat-square" alt="ECCV 2026">

<p>
  <a href="https://scholar.google.com/citations?user=PJqOR8gAAAAJ">Theodor Westny</a><sup>1</sup>&nbsp;·&nbsp;
  <a href="https://scholar.google.com/citations?user=pHcGdyEAAAAJ">David Axelsson</a><sup>1</sup>&nbsp;·&nbsp;
  <a href="https://scholar.google.com/citations?user=jxYDrDMAAAAJ">Björn Olofsson</a><sup>1,2</sup>&nbsp;·&nbsp;
  <a href="https://scholar.google.com/citations?user=o7sLRpcAAAAJ">Erik Frisk</a><sup>1</sup>
</p>

<sup>1</sup>Linköping University&nbsp;&nbsp;&nbsp;<sup>2</sup>Lund University

</div>

## Overview

The growing availability of trajectory datasets has fueled major advances in data-driven motion prediction.
Yet, models trained on one dataset often fail to generalize beyond their training domain as a result of differences in scene layouts, agent behaviors, and sensing conditions.
We present a framework that learns latent representations of datasets and quantifies their similarity using distributional metrics.
This large-scale study covers 24 major datasets — including the most widely used motion-prediction benchmarks — and shows that the resulting transferability scores strongly correlate with cross-dataset model performance.
The results provide practical guidance for dataset selection, pretraining, and large-scale foundation models for motion prediction.

<div align="center">
  <img src="https://westny.github.io/transferatlas/static/images/teaser.png" width="75%" alt="t-SNE projection of learned scenario embeddings">
  <br>
  <sub><b>t-SNE projection of learned scenario embeddings</b> (contours indicate density). Dataset proximity (e.g., ETH/inD, INTERACTION/WOMD) indicates potential for cross-dataset pretraining or knowledge transfer.</sub>
</div>

## Installation

TransferAtlas uses [uv](https://docs.astral.sh/uv/) and supports Python
3.10-3.13. Choose exactly one PyTorch backend when creating the environment:

```bash
# CPU-only PyTorch
uv sync --extra cpu

# PyTorch with CUDA 12.6
uv sync --extra cu126
```

The installed commands are:

```text
transferatlas-train
transferatlas-latents
transferatlas-kl
```

Use `uv run <command> --help` to inspect the arguments for each workflow.

## Data layout

TransferAtlas consumes trajectory scenes preprocessed with
[Dronalize](https://github.com/westny/dronalize). Set the dataset root in
`configs/trace.yml` or pass `--root` to a command. The expected directory layout
is:

```text
<root>/
  <dataset>/
    train/*.pkl
    val/*.pkl
    test/*.pkl
```

## Usage

A model checkpoint can be used to compute latent statistics for one
preprocessed dataset:

```bash
uv run transferatlas-latents \
  --config trace \
  --root /path/to/preprocessed-data \
  --dataset highD \
  --checkpoint artifacts/checkpoints/trace32.pt \
  --use-cuda true \
  --dry-run false
```

Statistics are written below `artifacts/latent-statistics/trace32/`. After
computing more than one dataset, create the directed KL matrix with:

```bash
uv run transferatlas-kl \
  --stats-dir artifacts/latent-statistics/trace32 \
  --output artifacts/latent-statistics/trace32/kl_divergence.csv
```

The numerical analyses included with the paper data run independently of the
raw trajectory datasets:

```bash
uv run python -m analysis.analyze_zeroshot
uv run python -m analysis.analyze_forgetting
```

## Reproduction scope

This repository provides the TRACE latent-embedding model, latent Gaussian and
KL computation, a trained checkpoint, and the two analyses above. It does not
include the third-party trajectory datasets, their complete preparation
workflow, the QCNet experiments, or scripts for every figure and ablation in the
paper.

The released latent statistics and KL results use covariance jitter
`alpha = 1e-4`. The `3e-3` value stated in Equation 6 of the paper is a typo.
`configs/trace.yml` records model parameters and training hyperparameters.

## Citation

```bibtex
@inproceedings{westny2026unveiling,
  title     = {Unveiling Transferability in Trajectory Prediction via Latent Scene Embeddings},
  author    = {Westny, Theodor and Axelsson, David and Olofsson, Bj{\"o}rn and Frisk, Erik},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

<sub>Project page template adapted from <a href="https://nerfies.github.io/">Nerfies</a>.</sub>

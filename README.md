<div align="center">

# Unveiling Transferability in Trajectory Prediction<br>via Latent Scene Embeddings

<p>
  <a href="https://arxiv.org/abs/2606.30777"><img src="https://img.shields.io/badge/arXiv-2606.30777-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/Paper-coming%20soon-lightgrey?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Paper coming soon">
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

The two profiles are mutually exclusive because `pyg-lib` must be compiled for
the selected PyTorch and CUDA combination. Re-run the chosen command after
switching profiles; uv will synchronize `.venv` with the checked-in lockfile.

The installed commands are:

```text
transferatlas-train
transferatlas-latents
transferatlas-kl
```

Use `uv run <command> --help` to inspect the arguments for each workflow.

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

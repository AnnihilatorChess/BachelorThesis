# Stabilizing Neural PDE Surrogates: Architecture and Autoregressive Regularization

> **Note:** This repository is currently a **Work In Progress (WIP)** and is actively undergoing changes. It contains the experimental codebase for a Bachelor's Thesis at Johannes Kepler University (JKU) Linz. The project focuses on mitigating autoregressive distribution shifts (error accumulation) in neural surrogate models for transient partial differential equations (PDEs). It builds heavily upon [The Well](https://polymathic-ai.org/the_well/) benchmark library and serves as a direct continuation of the author's prior university practical work.

## About This Repository

This repository hosts the code for my ongoing Bachelor's Thesis, which investigates methods to enhance the long-term rollout stability and physical parameter conditioning of neural PDE solvers.

The foundational codebase was established during my prior practical work, where stability interventions were evaluated on the `turbulent_radiative_layer_2D` dataset. The current thesis extends this research to systematically benchmark various architectures and autoregressive regularization techniques across more complex, long-horizon transient datasets.

## Project Overview

The core objective is to evaluate and improve how neural surrogate models handle distribution shifts during autoregressive generation.

The project explores:

### Surrogate Architectures

Evaluating global, pseudo-global, and local operators, including:

- Fourier Neural Operators (FNO)
- Classical U-Net
- ConvNeXt U-Net  
- Planned: Convolutional Neural Operators

### Conditioning Mechanisms

Adapting models to varying physical parameters (e.g., cooling time `t_cool`) via:

- **Naive Input Conditioning (IC)**
- **Embedded Input Conditioning (EIC)**
- **Feature-wise Linear Modulation (FiLM) layers**

### Autoregressive Training Stability

Implementing strategies to enforce zero-stability and damp self-generated errors, specifically:

- **Pushforward Trick** (unrolled training)
- Ongoing research into:
  - Temporal bundling
  - Latent jittering

## Installation

The codebase relies on the environment setup from **The Well**.

Clone the repository:

```bash
git clone https://github.com/AnnihilatorChess/PracticalWorkAI.git
cd PracticalWorkAI
```

It is recommended to use a Conda environment. Install Python dependencies:

```bash
pip install -e .
```

### Download the Dataset

Datasets can be downloaded using the provided CLI tool. Example:

```bash
the-well-download --base-path path/to/base --dataset turbulent_radiative_layer_2D
```

## Usage

The main training script is located in:

```
the_well/benchmark
```

All experiments are configured using **Hydra**. Configuration files are located in:

```
the_well/benchmark/configs
```

### 1. Running Baseline Models

No conditioning and standard one-step training:

```bash
cd the_well/benchmark
python train.py experiment=fno server=local data=turbulent_radiative_layer_2D
```

### 2. Running FiLM-Conditioned Models

FiLM-conditioned variants support varying physical parameters.

Available models:

- `fno_film`
- `unet_classic_film`
- `unet_convnext_film`

Example training command:

```bash
python train.py experiment=fno_film trainer.film=True data=turbulent_radiative_layer_2D server=local
```

The `film=True` flag ensures the data loader and model use conditioning (time and `t_cool`).

### 3. Training with the Pushforward Trick

Enable unrolled autoregressive training for improved stability:

```bash
python train.py experiment=fno trainer.pushforward=True data=turbulent_radiative_layer_2D server=local
```

## Key Modifications & Contributions

The following additions and fixes were integrated into the original **The Well** codebase during the preceding practical work and form the foundation for this thesis.

### Architecture Modifications (`models/`)

- **FiLM Integration**  
  New model variants with `_film` suffix for FNO, U-Net, and ConvNeXt U-Net.

- **Conditioning Logic**  
  Updated forward passes to accept scalar physical inputs (e.g., time, `t_cool`).

- **Conditioning Complexity Tiers**

  1. Naive input concatenation
  2. Embedded input concatenation
  3. FiLM layer integration

### Training Logic (`train.py`, `trainer/training.py`)

- **Pushforward Training Loop**  
  Modified training and data loading to support unrolled autoregressive sequences.

- **Conditioning Handlers**  
  Added helper logic to route physical parameters correctly during standard and unrolled training.

- **Model Checkpointing**  
  Updated checkpoint logic to save the model with the best validation **VRMSE (`val_VRMSE`)**.

### Upstream Bug Fixes to *The Well*

Several bugs in the benchmark were identified and reported upstream.

- **Parameter Counting Mistake**  
  FNO had fewer parameters than intended.  
  https://github.com/PolymathicAI/the_well/issues/67

- **Incorrect Logging of Long-Term Metrics**  
  Rollout metrics were overwritten every batch instead of averaged over the epoch.  
  https://github.com/PolymathicAI/the_well/issues/78

- **Incorrect Implementation of FNO and TFNO**  
  Spectral blocks were bypassed.  
  https://github.com/PolymathicAI/the_well/pull/64

- **Learning Rate Scheduler State Not Saved**  
  Scheduler state was not included in checkpoints.  
  https://github.com/PolymathicAI/the_well/pull/63

- **Best Model Weights Overwritten**  
  Validation checkpoints were overwritten each validation step.  
  https://github.com/PolymathicAI/the_well/pull/60

### Configuration Updates (`configs/`)

- Added new trainer flags:

```
film: bool
pushforward: bool
```

- Added experiment configs for FiLM models:

```
model/fno_film.yaml
```

- Added pushforward training hyperparameters to control:

- unroll depth
- curriculum schedules

## Acknowledgements and Attribution

This project is built upon **The Well**, a large-scale dataset collection for physics simulations.

- Original repository:  
  https://github.com/PolymathicAI/the_well

- Paper:  
  https://arxiv.org/abs/2412.00568

If you use this code, please cite the original work:

```bibtex
@article{ohana2024well,
  title={The well: a large-scale collection of diverse physics simulations for machine learning},
  author={Ohana, Ruben and McCabe, Michael and Meyer, Lucas and Morel, Rudy and Agocs, Fruzsina and Beneitez, Miguel and Berger, Marsha and Burkhart, Blakesly and Dalziel, Stuart and Fielding, Drummond and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={44989--45037},
  year={2024}
}
```
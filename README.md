# Enhancing Stability and Conditioning in Neural Surrogates for Turbulent Radiative Layers

This repository contains the code for a university practical work on Neural Surrogate Modeling for Turbulent Radiative Layers at JKU. It is built upon [The Well](https://polymathic-ai.org/the_well/), a collection of datasets for physical simulations including a framework for benchmarking neural PDE solvers. This project extends this work with modifications designed for improving stability on the `turbulent_radiative_layer_2D` dataset.

## Project Overview

The goal of this project is to evaluate different conditioning and training strategies for neural surrogate models. The project focuses on:

- Surrogate architectures: Fourier Neural Operators (FNO), Classical U-Net (U-Net) and ConvNext U-Net (CNextU-Net)
- Conditioning mechanisms on physical scalar parameters cooling time ($t_{cool}$) and simulation time:
  - naive input conditioning
  - embedded input conditioning
  - integration of Feature-wise Linear Modulation (FiLM) layers to condition models 
- Training stability: implementation of the Pushforward Trick to improve autoregressive prediction stability

## Installation

The codebase relies on the environment setup from `the_well`.

Clone the repository:

```cmd
git clone https://github.com/AnnihilatorChess/PracticalWorkAI.git
cd PracticalWorkAI
```

It is recommended to use a Conda environment. Install Python dependencies:

```cmd
pip install -e .
```

### Download the dataset

This project uses the `turbulent_radiative_layer_2D` dataset.


```cmd
the-well-download --base-path path/to/base --dataset turbulent_radiative_layer_2D
```

## Usage

The main training script is located in `the_well/benchmark`. All experiments are configured using Hydra. The configuration files can be found in `the_well/benchmark/configs`

1. Running baseline models (no FiLM, no Pushforward)

```cmd
cd the_well/benchmark
python train.py experiment=fno server=local data=turbulent_radiative_layer_2D
```


2. Running FiLM-conditioned models

FiLM-conditioned variants have been implemented to handle varying physical parameters.

Available FiLM models:
- `fno_film`
- `unet_classic_film`
- `unet_convnext_film`

Train a FiLM model (ensure `film=True` so the data loader and model use conditioning, this can be set in the `configs/trainer/defaults` or passed directly into the command):

```cmd
python train.py experiment=fno_film trainer.film=True data=turbulent_radiative_layer_2D server=local
```

The `film=True` flag ensures the data loader augments input channels with time and `t_cool`.

3. Training with the Pushforward Trick
Set `pushforward=True` in 'configs/trainer/defaults' or pass the parameter directly into the command:

```cmd
python train.py experiment=fno trainer.pushforward=True data=turbulent_radiative_layer_2D server=local
```

## Key Modifications

Key changes made to the original `the_well` codebase to support this research:

### Architecture modifications (`models/`)

- FiLM layers: new model variants suffixed with `_film` for FNO, U-Net, and CNextU-Net
- Conditioning logic: forward passes accept scalar inputs (time, `t_cool`)
- 3 different options for conditioning complexity:
  - naive input conditioning
  - naive input conditioning with embedding
  - FiLM Layer integration into model architecture

### Training logic (`train.py`, `trainer/training.py`)

- Pushforward training loop in `training.py`. Adjusted train data loaders
- Adjusted training loop to handle conditioning. Added several helper functions.
- Model checkpointing: checkpointing logic modified to save the model with the best `val_VRMSE`.

### Bug fixes

- **Parameter counting mistake**  
  Caused the FNO to have fewer parameters than intended. This was discovered by another contributor in  
  https://github.com/PolymathicAI/the_well/issues/67

- **Incorrect logging of long-term metrics**  
  The metrics used for plotting Figures 1–4 were overwritten every batch instead of being averaged over the full epoch. We reported this in  
  https://github.com/PolymathicAI/the_well/issues/78

- **Incorrect implementation of FNO and TFNO**  
  The spectral blocks in both model implementations were not used. We opened the following pull request, which was merged the same day:  
  https://github.com/PolymathicAI/the_well/pull/64  
  In this PR, we also noted that the test set was evaluated using the most recent model weights instead of the best validation checkpoint.

- **Learning rate scheduler state not saved**  
  The scheduler state was not stored, leading to inconsistent learning rate behavior when resuming training. Addressed in:  
  https://github.com/PolymathicAI/the_well/pull/63

- **Best model weights overwritten**  
  The best validation checkpoint was overwritten at each validation step. Identified alongside:  
  https://github.com/PolymathicAI/the_well/pull/60

### Configuration (`configs/`)

- New flags: added `film` and `pushforward` booleans to `trainer/defaults.yaml` to toggle these modes.
- Configuring FiLM models: added experiment configs (e.g. model/fno_film.yaml`), here the type of conditioning and conditioning parameters can be set
- Configuring Pushforward: added parameters to modify pushforward behavior in  `trainer/defaults`

## Acknowledgements and Attribution

This project is built upon The Well, a large-scale dataset for physics simulations. We acknowledge the authors for providing the framework, dataset, and baseline implementations. For a more general explanation of the underlying codebase and configurations please refer to the original repository.

- Original repository: [PolymathicAI/the_well](https://github.com/PolymathicAI/the_well)  
- [Paper](https://arxiv.org/abs/2412.00568)


If you find this code useful, please cite the original work:

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

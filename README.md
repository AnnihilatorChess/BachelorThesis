# Stabilizing Autoregressive Neural PDE Surrogates

Codebase and thesis source for the Bachelor's Thesis **"Stabilizing Autoregressive Neural PDE Surrogates: An Ablation and Capacity Trade-off Study"** (Johannes Kepler University Linz).

This repository evaluates lightweight stabilization techniques (Temporal Bundling, Pushforward, Noise Injection, and their combination) across diverse PDE datasets and architectures to prevent autoregressive error accumulation. The central findings are that these techniques reliably improve rollout accuracy at close to one-step training cost, and that the memory they save relative to full Backpropagation Through Time (BPTT) can be reinvested into larger models that outperform the VRAM-constrained models BPTT is restricted to on the simpler PDEs.

## Fork & Attribution

This repository is a fork of the official [The Well](https://github.com/PolymathicAI/the_well) benchmark and integrates evaluation methodologies, model scales, and datasets from [PDEBench](https://github.com/pdebench/PDEBench). We gratefully acknowledge the authors of both benchmarks for open-sourcing their tools and data.

### References and Citations

- **The Well**: Ohana et al., "The Well: A Large-Scale Collection of Diverse Physics Simulations for Machine Learning", NeurIPS 2024.  
  [Paper](https://arxiv.org/abs/2412.00568) | [Repository](https://github.com/PolymathicAI/the_well)
  
- **PDEBench**: Takamoto et al., "PDEBench: An Extensive Benchmark for Scientific Machine Learning", NeurIPS 2022.  
  [Paper](https://arxiv.org/abs/2210.07182) | [Repository](https://github.com/pdebench/PDEBench)

---

## Datasets and Architectures

### Evaluated Datasets
The experimental stack bridges chaotic physical systems from The Well and shock-forming equations from PDEBench (converted to The Well's unified HDF5 format):
- **The Well**: `turbulent_radiative_layer_2D`, `active_matter`
- **PDEBench**: `pdebench_swe` (Shallow Water Equations), `pdebench_1d_burgers` (1D Burgers, downsampled)

### Evaluated Architectures
- **Fourier Neural Operator (FNO)** (Global spectral processing)
- **U-Net Classic** (Local convolutional processing)
- **Convolutional Neural Operator (CNO)** (Continuous convolutional operator blocks)

---

## Installation

The environment mirrors The Well's original setup. We highly recommend using a Conda environment.

```bash
git clone https://github.com/AnnihilatorChess/BachelorThesis.git
cd BachelorThesis
pip install -e ".[benchmark]"
```

The `benchmark` extra pulls in the training dependencies (Hydra, Weights & Biases, neuraloperator, timm, torch_harmonics). Python 3.10+ is required.

> **Note on CNO:** It is highly recommended to use the compiled version of the Convolutional Neural Operator (CNO) for optimal performance. Compiling the CNO requires additional system-level libraries. Please refer to the official [PDEBench repository](https://github.com/pdebench/PDEBench) for more information and detailed installation instructions.

### Dataset Preparation

**The Well Datasets:**
Native datasets can be downloaded using The Well's CLI tool:
```bash
the-well-download --base-path datasets/ --dataset turbulent_radiative_layer_2D
```

**PDEBench Datasets:**
PDEBench datasets are not distributed in The Well's format, so they must be downloaded from the [official PDEBench data repository](https://github.com/pdebench/PDEBench) and converted into The Well's unified HDF5 layout before training. This repository ships two self-contained conversion scripts for that (they only require `h5py`, `numpy`, and `pyyaml`).

*Shallow Water Equations:* download the radial-dam-break file `2D_rdb_NA_NA.h5`, place it at `datasets/PDEBench/2D_SWE/2D_rdb_NA_NA.h5`, then run:
```bash
python scripts/convert_pdebench_swe.py        # writes datasets/pdebench_swe/
```

*1D Burgers (low viscosity):* download `1D_Burgers_Sols_Nu0.001.hdf5`, place it at `datasets/PDEBench/1D_Burgers/1D_Burgers_Sols_Nu0.001.hdf5`, then run:
```bash
python scripts/convert_pdebench_1d_burgers.py             # full resolution -> datasets/pdebench_1d_burgers/
python scripts/convert_pdebench_1d_burgers.py --variant pdebench   # subsampled 4x in space, 5x in time -> datasets/pdebench_1d_burgers_pdebench/
```

Both scripts create the `data/{train,valid,test}/` split layout and a `stats.yaml` of normalization statistics that the dataloaders expect. The thesis uses the PDEBench-protocol (downsampled) Burgers variant to stay comparable to PDEBench. Pass `--help` to either script for the splitting and subsampling options.

---

## Important Configuration Parameters

All experiments are driven by **Hydra**. The configurations are deeply nested, but the most important parameters to control the ablation stack are:

| Parameter | Description |
|---|---|
| `experiment=` | Selects the model and dataset baseline (e.g., `fno`, `unet_classic`, `cno`). Use `_pdebench` variants for the downscaled "small" models. |
| `data=` | Selects the dataset to train on (e.g., `turbulent_radiative_layer_2D`, `pdebench_swe`). |
| `temporal_bundle_size=` | Set to > 1 to predict multiple timesteps in a single forward pass (Temporal Bundling). Default is 1. |
| `trainer.pushforward=` | `True` enables unrolled autoregressive training with curriculum learning. |
| `trainer.noise_injection=` | `True` adds Gaussian noise to inputs. Must be used with `trainer.noise_std=...` |
| `trainer=bptt` | Switches the trainer to full Backpropagation Through Time (use only with the small models). |
| `seed=` | Sets the global RNG seed for reproducibility. |

---

## Usage

All training experiments are managed via **Hydra** and execute from the `the_well/benchmark` directory.

### 1. Standard Training (One-Step Baseline)
Train a standard 1-step autoregressive model without stabilization:
```bash
cd the_well/benchmark
python train.py experiment=fno server=local data=turbulent_radiative_layer_2D
```

### 2. Stabilization Techniques (Ablation Stack)
You can enable various stabilization techniques to prevent error accumulation on long rollouts by passing Hydra overrides:

- **Temporal Bundling (e.g., K=4):** Predict multiple timesteps simultaneously.
  ```bash
  python train.py experiment=fno server=local data=turbulent_radiative_layer_2D temporal_bundle_size=4
  ```

- **Pushforward Trick:** Unrolled autoregressive training without backpropagating through intermediate targets.
  ```bash
  python train.py experiment=fno server=local data=turbulent_radiative_layer_2D trainer.pushforward=True
  ```

- **Noise Injection:** Gaussian noise data augmentation.
  ```bash
  python train.py experiment=fno server=local data=turbulent_radiative_layer_2D trainer.noise_injection=True trainer.noise_std=0.01
  ```

- **Combined Synergy (TB + PF):**
  ```bash
  python train.py experiment=fno server=local data=turbulent_radiative_layer_2D temporal_bundle_size=4 trainer.pushforward=True
  ```

### 3. BPTT on PDEBench Models
To replicate the PDEBench training methodology (using heavily downscaled "small" models so full Backpropagation Through Time fits into VRAM):
```bash
python train.py experiment=fno_pdebench server=local data=pdebench_swe trainer=bptt
```
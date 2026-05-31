# Stabilizing Autoregressive Neural PDE Surrogates

Codebase for the Bachelor Thesis: **"Stabilizing Autoregressive Neural PDE Surrogates: Overcoming the Capacity-Stability Trade-off via Lightweight Regularization"** (Johannes Kepler University Linz).

This repository evaluates lightweight stabilization techniques (Temporal Bundling, Pushforward, Noise Injection) across diverse PDE datasets to prevent autoregressive error accumulation. The core contribution demonstrates that these stabilization methods allow for the training of large, highly expressive models that significantly outperform VRAM-constrained models trained with full Backpropagation Through Time (BPTT).

## Fork & Attribution

This repository is a fork of the official [The Well](https://github.com/PolymathicAI/the_well) benchmark and integrates evaluation methodologies, model scales, and datasets from [PDEBench](https://github.com/pdebench/PDEBench). We gratefully acknowledge the authors of both benchmarks for open-sourcing their incredible tools and data.

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
- **PDEBench**: `pdebench_swe` (Shallow Water Equations), `pdebench_1d_burgers_pdebench` (1D Burgers, downsampled)

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
pip install -e .
```

> **Note on CNO:** It is highly recommended to use the compiled version of the Convolutional Neural Operator (CNO) for optimal performance. Compiling the CNO requires additional system-level libraries. Please refer to the official [PDEBench repository](https://github.com/pdebench/PDEBench) for more information and detailed installation instructions.

### Dataset Preparation

**The Well Datasets:**
Native datasets can be downloaded using The Well's CLI tool:
```bash
the-well-download --base-path datasets/ --dataset turbulent_radiative_layer_2D
```

**PDEBench Datasets:**
PDEBench datasets must be downloaded from their [official repository/website](https://github.com/pdebench/PDEBench). Once downloaded, use the provided conversion scripts to format them into The Well's unified HDF5 structure:
```bash
python scripts/convert_pdebench_swe.py --input_path path/to/swe.h5 --output_path datasets/pdebench_swe
```
*(Check the `scripts/` directory for other conversion utilities like `convert_pdebench_1d_burgers.py`)*

---

## Important Configuration Parameters

All experiments are driven by **Hydra**. The configurations are deeply nested, but the most important parameters to control the ablation stack are:

| Parameter | Description |
|---|---|
| `experiment=` | Selects the model and dataset baseline (e.g., `fno`, `unet_classic`, `cno`). Use `_pdebench` variants for the downscaled "Tiny" models. |
| `data=` | Selects the dataset to train on (e.g., `turbulent_radiative_layer_2D`, `pdebench_swe`). |
| `temporal_bundle_size=` | Set to > 1 to predict multiple timesteps in a single forward pass (Temporal Bundling). Default is 1. |
| `trainer.pushforward=` | `True` enables unrolled autoregressive training with curriculum learning. |
| `trainer.noise_injection=` | `True` adds Gaussian noise to inputs. Must be used with `trainer.noise_std=...` |
| `trainer=bptt` | Switches the trainer to Full Backpropagation Through Time (use only with Tiny models). |
| `seed=` | Sets the global RNG seed for reproducibility. |

---

## Usage

All training experiments are managed via **Hydra** and execute from the `the_well/benchmark` directory. 

### 1. Standard Training (Our Baseline)
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
To replicate the PDEBench training methodology (using heavily downscaled "Tiny" models to fit full Backpropagation Through Time into VRAM):
```bash
python train.py experiment=fno_pdebench server=local data=pdebench_swe trainer=bptt
```

---

## Documentation and Analysis

For detailed insights into the experimental design, performance metrics, and automated evaluation aggregations, please refer to the `docs/` folder:

- `docs/thesis-outline.md` - The core narrative, methodology, and outline of the Bachelor Thesis.
- `docs/performance_analysis/` - Detailed performance metrics, aggregated directly from Weights & Biases across multiple seeds.
- `docs/wandb_analysis.md` - Guidelines and conventions for dataset evaluation, checkpoint selection strategies (One-Step vs. Rollout weights), and metric interpretation.

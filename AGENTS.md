# AGENTS.md

Shared guidance for agentic systems working in this repository (Claude Code, Gemini CLI, and any other coding agent).

## Project Overview

Bachelor's thesis at JKU Linz extending [The Well](https://github.com/PolymathicAI/the_well) benchmark for neural PDE surrogates. The goal is stabilizing autoregressive rollout via **pushforward training, temporal bundling, and noise injection** across multiple architectures (FNO, U-Net Classic, U-Net ConvNeXt, CNO) and datasets.

**CRITICAL:** The `docs/` directory contains essential technical specifications, metric definitions, and baseline results. Always consult these files before proposing or implementing changes.

## Multi-Agent Environment

This repository is managed by multiple agentic systems (Claude Code, Gemini CLI, etc.). This file is the single source of truth — both `CLAUDE.md` and `GEMINI.md` are thin pointers to it.

### Collaboration Guidelines
- **Conflict avoidance:** Always check `git status` and recent logs before making changes.
- **Shared standards:** Adhere strictly to the coding style and commands defined here.
- **Communication:** If a strategy changes, update this file (not the pointer files).

## Setup

- **Python environment:** All commands MUST be run using the `bsc_` conda environment located at `C:\Users\simon\miniconda3\envs\bsc_`.
- Use the full path for the Python interpreter (e.g., `C:\Users\simon\miniconda3\envs\bsc_\python.exe`) to ensure consistency.

```bash
pip install -e ".[benchmark,dev]"
```

The `benchmark` extra includes hydra-core, wandb, neuraloperator, timm, and torch_harmonics. The `dev` extra adds pytest and ruff. Python 3.10+ required.

## Common Commands

**Lint / format:**
```bash
ruff check .          # lint
ruff format .         # auto-format
```

**Tests** (no test suite exists yet; pytest is in dev deps for future use):
```bash
pytest
```

**Quick debug run** (end-to-end sanity check — tiny FNO, 1 epoch, hardcoded to `active_matter` data; the `data=` override has no effect here):
```bash
python the_well/benchmark/train.py experiment=debug server=local
```

**Fast iteration on a specific dataset** (real config, but exits quickly):
```bash
python the_well/benchmark/train.py experiment=fno server=local data=pdebench_swe \
  trainer.epochs=2 trainer.rollout_val_frequency=999
```

**Download datasets:**
```bash
the-well-download --dataset turbulent_radiative_layer_2D --base_path ./datasets
```

**Create miniwell dataset** (downsampled gray_scott for fast local testing):
```bash
python run_miniwell.py
```

**Convert PDEBench datasets to The Well format:**
```bash
python scripts/convert_pdebench_swe.py
python scripts/convert_pdebench_1d_burgers.py
```

## Training

Training is driven by [Hydra](https://hydra.cc/) config composition. The three required overrides are `experiment`, `server`, and `data`:

```bash
python the_well/benchmark/train.py experiment=fno server=local data=turbulent_radiative_layer_2D
```

Key optional overrides:
```bash
trainer.pushforward=True                    # Unrolled autoregressive training
temporal_bundle_size=4                      # Predict K timesteps per forward pass
trainer.noise_injection=True trainer.noise_std=0.01
```

Resume from checkpoint:
```bash
python the_well/benchmark/train.py experiment=fno server=local data=... +resume_path=/path/to/checkpoint
```

## Architecture

### Config system (`the_well/benchmark/configs/`)

Hydra merges: `config.yaml` (base) → `experiment/*.yaml` (preset) → `model/*.yaml` + `data/*.yaml` + `trainer/defaults.yaml`. Experiment configs are the main entry point combining model + data + trainer settings.

Available server configs: `local`, `rusty`, `student`, `student_1d`, `gpuxl` — each sets dataset base paths, batch sizes, and DDP flags for that compute environment.

### Data pipeline (`the_well/data/`)

- `datasets.py` — `WellDataset` loads HDF5 files; `DeltaWellDataset` predicts velocity deltas instead of absolute states.
- `datamodule.py` — `WellDataModule` creates train/val/test/rollout dataloaders.
- `data_formatter.py` — converts `[B, T, H, W, C]` ↔ model format `[B, T*C, H, W]`; `process_output_unbundle()` handles temporal bundling unpacking.

Batch tensors: `input_fields [B, T_in, H, W, C]`, `output_fields [B, T_out, H, W, C]`, `constant_scalars [B, 1]` (e.g., `t_cool`), `constant_fields [B, H, W, C_const]`.

Normalization stats are stored per-dataset in `datasets/<name>/stats.yaml` and loaded at runtime.

### Models (`the_well/benchmark/models/`)

All models inherit from `BaseModel` in `common.py`. **Active models for the thesis: `FNO`, `UNetClassic`, `UNetConvNeXt`, `CNO`.**

Temporal bundling: models output `[B, K*C, H, W]` where K = `temporal_bundle_size`; the formatter unbundles to `[B, K, H, W, C]`.

### Trainer (`the_well/benchmark/trainer/training.py`)

Two training modes:
- `train_one_epoch()` — standard single-step MSE.
- `train_one_epoch_pushforward()` — unrolled curriculum training; samples rollout depth 0–3 with probabilities ramping from `[1,0,0,0]` to `[0.4, 0.2, 0.2, 0.2]` over `pushforward_warmup_epochs`.

Rollout/validation: `rollout_model()` autoregressively generates full trajectories; metrics are split into temporal intervals (early/mid/late) and logged per-field to W&B. Extended metrics: `cRMSE` (PDEBench conservation error), `ValidRolloutLength`, `CorrelationTime`, `ErrorGrowthRate`.

Checkpoints (run output dir):
- `recent.pt` — full training state (model + optimizer + scheduler + `best_metrics` dict); used for resume.
- `best_one_step_vrmse.pt`, `best_one_step_nrmse.pt` — updated whenever the short-validation metric improves.
- `best_rollout_vrmse.pt`, `best_rollout_nrmse.pt` — updated whenever the full-rollout validation metric improves (every `rollout_val_frequency` epochs).

`best_*.pt` files contain weights only (plus `metric_name`, `metric_value`, `epoch`). Final test evaluation loads each `best_*.pt` in turn and logs `test_best_<metric>_*` / `rollout_test_best_<metric>_*` to W&B, plus `summary/<metric>__{test,rollout_test}_{vrmse,nrmse}` headline scores in `wandb.run.summary`.

## Thesis-Specific Constraints

- **Datasets:** `turbulent_radiative_layer_2D`, `pdebench_swe`, `gray_scott_reaction_diffusion`, `rayleigh_benard`, `pdebench_1d_burgers`.
- **Data strategy:** No spatial/temporal downsampling. Use full resolution but limit the number of trajectories for massive datasets (>100 GB).
- **Scope:** Focus is on architecture and stabilization techniques.
- **Stabilization:** Pushforward trick, noise injection, temporal bundling.

## Research & Documentation Mandate

- **Docs first:** Before proposing or implementing changes, search and read relevant files in the `docs/` directory. Start with `docs/INDEX.md`.
- **Empirical baselines:** Consult `docs/performance_analysis/summary.md` and `docs/wandb_analysis.md` for existing experimental results and metric interpretations.
- **Metric definitions:** `docs/extended-metrics.md` is the primary reference for rollout evaluation.

## Key Design Decisions

- **Distributed training**: DDP with `DistributedDataSampler`; metrics are all-reduced across ranks.
- **Mixed precision**: `float16` (with GradScaler) or `bfloat16`, controlled by trainer config.
- **Ruff** is the linter/formatter; notebooks are excluded from formatting but included in linting.

## Documentation Index

For comprehensive project knowledge, see **[docs/INDEX.md](docs/INDEX.md)**, which indexes:
- Project planning & thesis scope
- Data conversion guides
- W&B experiment tracking & metrics
- Development workflow

# PDEBench 2D Shallow Water Equations (SWE) Experiments

This document details the exact experimental setup used in the PDEBench paper and repository for the 2D Shallow Water Equations (SWE) dataset (`2D_rdb_NA_NA.h5`), and provides a blueprint for running comparable experiments in our codebase, including a recommendation for the CNO model.

## 1. PDEBench Experimental Setup for SWE

Based on a thorough inspection of the PDEBench paper and their official codebase (`pdebench/models/config/config_rdb.yaml` and `pdebench/models/fno/utils.py`), here is the exact training setup:

### Data Resolution & Downsampling
- **No Downsampling:** Unlike 1D Burgers, PDEBench **does not** downsample the 2D SWE dataset. 
- The models are trained on the full resolution: `128 x 128` spatial grid, with `101` time steps. 
- (Verified via `config_rdb.yaml` where `reduced_resolution: 1` and `reduced_resolution_t: 1`).
- Only **1 variable** (water height `h`) is used as input and output.

### Hyperparameters
- **Context Window (`initial_step`):** 10 steps.
- **Batch Size:** 5. (Note: Using `batch_size=64` is recommended for our hardware, but note that it increases the total parameter updates per epoch).
- **Epochs:** 500.
- **Learning Rate:** `1e-3` (If using batch size 64, consider tuning this to `3e-3`).
- **Optimizer:** Adam (with `weight_decay=1e-4`).
- **LR Scheduler:** StepLR (`step_size=100`, `gamma=0.5`).
- **Precision:** PDEBench uses full FP32 (no AMP observed in code or paper).
- **Validation Frequency:** Every 10 epochs.

### PDEBench Model Architectures
- **FNO:** Extremely small. `modes=12`, `width=20` (4 spectral layers). This results in only ~150,000 parameters.
- **U-Net:** `init_features=32`, resulting in exactly **~7.76 Million** parameters. U-Net is trained with `pushforward=True` and `unroll_step=20`.

---

## 2. Best Model Selection & Logging Strategy

### PDEBench's Strategy
- They evaluate the model every 10 epochs on the validation set by running a **full autoregressive rollout** (from step 10 to 101).
- They compute the overall normalized L2 error (which mathematically matches our `NRMSE` over the rollout).
- They save the checkpoint that achieves the lowest full-rollout error. 
- **Critical Flaw:** Their validation set and test set are the exact same 10% slice of the data. Their reported test metrics are from the epoch that performed best on that exact same data.

### Our Strategy (Recommendation)
Our current logging strategy saves 4 checkpoints:
1. `best_one_step_nrmse`
2. `best_one_step_vrmse`
3. `best_rollout_nrmse`
4. `best_rollout_vrmse`

**Recommendation: Keep all 4 models.** 
Our `best_rollout_nrmse` is the **exact equivalent** of PDEBench's model selection criterion. However, our methodology is much more rigorous because we maintain a strict separation between one-step metrics (how well the model learns the immediate dynamics) and rollout metrics (how well the model prevents error accumulation). By keeping all 4, we can provide a much deeper analysis of the trade-offs than PDEBench did.

Additionally, tracking **cRMSE (Conservation RMSE)** is highly recommended, as it is a primary metric in PDEBench for evaluating whether the model respects the underlying physics (mass conservation in SWE). 

---

## 3. The Runtime vs. Parameter Count Trade-off in PDEBench

A critical finding from analyzing the PDEBench paper (Table 15) is that they prioritize **Compute Parity over Capacity Parity**. Despite their U-Net having ~7.76M parameters and their FNO having only ~150K parameters, both models took almost exactly the same time to train per epoch (~100s on a V100 GPU). The FNO's global FFT operations are mathematically much more expensive per parameter than the U-Net's local convolutions.

When comparing your CNO results to PDEBench's FNO, you must acknowledge that you are comparing a model with drastically more memory capacity to one that was scaled down to train quickly.

---

## 4. CNO Parity Configuration Recommendation

Our default CNO model (`channel_multiplier: 96, N_layers: 3, N_res: 4, N_res_neck: 4`) has roughly ~18 Million parameters. 

To create a fair "PDEBench-Parity" CNO experiment, we should scale the CNO architecture down to roughly match the capacity of their U-Net (~7.76M params). By reducing the `channel_multiplier` to 64, we bring the parameter count down to ~7.9M.

**Recommended `cno_pdebench.yaml` Config (~8M Params):**
```yaml
_target_: the_well.benchmark.models.CNO
N_layers: 3
N_res: 4
N_res_neck: 4
channel_multiplier: 64
use_bn: True
```

**Recommended `experiment/cno_pdebench_swe.yaml` Config:**
```yaml
defaults:
  - /data: pdebench_swe
  - /model: cno_pdebench
  - /lr_scheduler: step_lr
  - override /optimizer: adam_pdebench
  - _self_

trainer:
  loss_fn:
    _target_: the_well.benchmark.metrics.MSE
  formatter: channels_first_default
  epochs: 500
  rollout_val_frequency: 10

data:
  n_steps_input: 10
  batch_size: 5

name: cno_pdebench_swe
```

By using this setup, you isolate the architectural differences (CNO vs. U-Net) while holding the parameter count, learning rate schedule, context window, and batch size constant against the PDEBench baselines.

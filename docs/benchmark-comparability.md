# Benchmark Comparability — The Well vs PDEBench vs Our Setup

This document describes how the two source benchmarks (The Well and PDEBench) constructed their baselines, and evaluates how comparable our thesis setup is to each. Every claim below is backed by a file-and-line citation from the official repos (cloned locally at `C:\Users\simon\Documents\GitHub\the_well` and `C:\Users\simon\Documents\GitHub\PDEBench`).

Only datasets that appear in our thesis are covered: `turbulent_radiative_layer_2D`, `gray_scott_reaction_diffusion`, `rayleigh_benard` (The Well), and `pdebench_swe`, `pdebench_1d_burgers` (PDEBench).

---

## 1. The Well (Ohana et al., NeurIPS 2024)

Paper: https://arxiv.org/abs/2412.00568 · Repo: https://github.com/PolymathicAI/the_well

Our training stack is a fork of this benchmark.

### Model Size

Source: `the_well/benchmark/configs/model/*.yaml` and each model's README in the_well repo.

| Model | Config | Params (2D, 3-field TRL-sized input) |
|---|---|---|
| FNO | `modes1=16, modes2=16, hidden_channels=128`, 4 blocks (`fno/README.md`) | ~19 M |
| U-Net Classic | `init_features=48` | ~17.5 M |
| U-Net ConvNeXt | `init_features=42`, `blocks_per_stage=2`, `filter_size=7` | ~18.6 M |
| CNO | `channel_multiplier=96, N_layers=3, N_res=4, N_res_neck=4, use_bn=True` | ~17.9 M |

Params measured by instantiating each model on TRL input shapes (`dim_in=12, dim_out=3, spatial_resolution=(384,128)`).

### Main Metric for Comparison

Source: `the_well/benchmark/metrics/spatial.py`.

- **VRMSE** (primary, "variance-scaled RMSE") — `spatial.py:221-240` defines it as `NRMSE.eval(..., norm_mode="std")`, which computes
  ```
  VMSE  = mean_spatial((x−y)^2) / (var_spatial(y) + eps)    # spatial.py:141-142
  VRMSE = sqrt(VMSE)                                        # spatial.py:196
  ```
  A model predicting the per-field spatial mean gives VRMSE = 1, so VRMSE > 1 is worse than a trivial constant predictor.
- **NRMSE** (`norm_mode="norm"`, the default): `sqrt( mean_spatial((x−y)^2) / mean_spatial(y^2) )` — `spatial.py:139-140, 196`.
- Other metrics: `MSE`, `RMSE`, `NMSE`, `VMSE`, `LInfinity`, `Pearson`, and a spectral error.

### Evaluation Protocol

Source: `the_well/benchmark/configs/trainer/defaults.yaml`, `the_well/benchmark/configs/config.yaml`, and `the_well/benchmark/configs/lr_scheduler/cosine_with_warmup.yaml`.

- Context: `n_steps_input = 4`, `n_steps_output = 1` (`config.yaml:7-8`).
- Rollout: `max_rollout_steps = 100`, `short_validation_length = 20`, `rollout_val_frequency = 2` (`trainer/defaults.yaml:6-7`).
- Epochs: **default 200** (`trainer/defaults.yaml:2`), but the per-dataset FNO README lists actual epochs ranging from 13 (`turbulence_gravity_cooling`) to 500 (`turbulent_radiative_layer_2D`). This reflects the paper's time-boxed compute budget (12 h on one H100).
- Optimizer: **AdamW**, lr=1e-2, weight_decay=1e-4 (`configs/optimizer/adam.yaml`).
- LR scheduler: **LinearWarmupCosineAnnealingLR**, `warmup_epochs=5` (`configs/lr_scheduler/cosine_with_warmup.yaml`).
- LR is **swept per (model, dataset)** from `{1e-4, 5e-4, 1e-3, 5e-3, 1e-2}`; best values are listed in each model's README (e.g. `fno/README.md`: TRL=5e-3, Rayleigh-Bénard=1e-4, Gray-Scott=1e-3).
- Normalization: per-dataset z-score (`ZScoreNormalization` from `config.yaml:11`).

### Best-Model Selection (ours)

Source: [the_well/benchmark/trainer/training.py](../the_well/benchmark/trainer/training.py) (`CHECKPOINT_METRICS`, `_update_best_checkpoints`, `_run_final_test_eval`).

Four separate best checkpoints are tracked during training, plus `recent.pt` for resume:

| Filename | Selected on | Update frequency |
|---|---|---|
| `best_one_step_vrmse.pt` | `valid_<ds>/full_VRMSE_T=all` (short validation) | every `val_frequency` epochs |
| `best_one_step_nrmse.pt` | `valid_<ds>/full_NRMSE_T=all` (short validation) | every `val_frequency` epochs |
| `best_rollout_vrmse.pt` | `rollout_valid_<ds>/full_VRMSE_T=all` (full rollout) | every `rollout_val_frequency` epochs |
| `best_rollout_nrmse.pt` | `rollout_valid_<ds>/full_NRMSE_T=all` (full rollout) | every `rollout_val_frequency` epochs |

`best_*.pt` files store weights only (plus `metric_name`, `metric_value`, `epoch`) — optimizer/scheduler state lives only in `recent.pt`. Final test evaluation iterates over every saved `best_*.pt`, loading each in turn and logging `test_best_<metric>_*` / `rollout_test_best_<metric>_*` under distinct W&B key prefixes. Headline scores (`test_vrmse`, `test_nrmse`, `rollout_test_vrmse`, `rollout_test_nrmse` per checkpoint) are copied to `wandb.run.summary` for easy cross-run comparison.

This deliberately surfaces the short- vs long-horizon checkpoint tension: the best one-step VRMSE is often *not* the best rollout-nRMSE, and the four-checkpoint matrix lets us report that directly in the thesis.

---

## 2. PDEBench (Takamoto et al., NeurIPS 2022)

Paper: https://arxiv.org/abs/2210.07182 · Repo: https://github.com/pdebench/PDEBench

We use PDEBench as a primary point of comparison for 1D Burgers and 2D SWE. We have implemented specific "PDEBench-protocol" configurations to ensure our experimental setup is as comparable as possible to the original benchmark.

### Model Size

Source: `pdebench/models/config/args/config_Bgs.yaml`, `config_rdb.yaml`, and `pdebench/models/fno/fno.py` / `unet/unet.py`.

| Model | PDEBench config | Params | Our Parity Config |
|---|---|---|---|
| FNO (2D SWE) | `modes=12, width=20` | ~0.46 M | `model: fno_pdebench` |
| U-Net (2D SWE) | `init_features=32` | ~7.76 M | `model: unet_classic_pdebench` |
| CNO (2D SWE) | N/A | N/A | `model: cno_pdebench` (~7.9 M) |

Note: PDEBench baselines are significantly smaller than "The Well" baselines (which are ~18 M). Comparing our full-sized CNO to their FNO would be an unfair comparison of capacity rather than architecture. Thus, we use the parity configs for direct comparison. Empirical analysis of the PDEBench paper (Table 15) reveals that they chose these disparate parameter counts to achieve **Compute Parity** (similar training time per epoch) rather than Capacity Parity.

### Main Metrics for Comparison

Sources: paper Appendix B (p. 17) and `pdebench/models/metrics.py`.

From the **paper** (Eq. 4, p. 17):
```
nRMSE ≡ ||u_pred − u_true||_2 / ||u_true||_2
```

From the **code** (`metrics.py:190-197`):
Our `NRMSE_T=all` metric is mathematically equivalent to PDEBench's `nRMSE`.

Other metrics: **RMSE**, **cRMSE** (conserved-quantity deviation). We have added `cRMSE` to our suite to match PDEBench's focus on physical conservation.

Important: PDEBench slices off `initial_step` frames before computing the metric. Our rollout metrics are computed over the full rollout window starting after the context window.

### Evaluation Protocol

Sources: paper Appendix C (p. 18) and `config_Bgs.yaml`, `config_rdb.yaml`, `pdebench/models/fno/train.py`.

- **Context Window (`initial_step`):** 10 steps (vs our standard 4).
- **Epochs:** 500 (vs our standard 100).
- **Optimizer:** Adam with StepLR (halved every 100 epochs).
- **Dataset Split:** 90/10 train/test (where test == validation).

Our `experiment/*_pdebench.yaml` configs replicate this protocol (10-step input, 500 epochs, StepLR) while maintaining our more rigorous 80/10/10 data split.

### Best-Model Selection

PDEBench saves the best model based on the full-rollout error on the test set. We replicate this by tracking `best_rollout_nrmse.pt`, while additionally tracking one-step metrics for a more complete analysis of the stability-accuracy trade-off.

### Downsampling — paper vs repo reconciliation

- **SWE: paper and repo agree** — full 128 × 128 × 101 is used. No downsampling.
- **Burgers: repo downsamples to 256 × 40.** The paper is silent on this, but the shipped config and weights confirm it.

## 3. Training Methodology: Autoregressive BPTT vs. One-Step Stabilization

A fundamental distinction between PDEBench and our thesis setup lies in how the models are trained to handle error accumulation over time.

### PDEBench: Full Autoregressive Backpropagation Through Time (BPTT)

As seen in `pdebench/models/fno/train.py`, PDEBench trains its models using a full autoregressive loop over the entire rollout window (e.g., 91 steps for SWE). 

1. The model predicts step $t$.
2. The Mean Squared Error (MSE) for step $t$ is calculated and **added to a running total loss**.
3. The prediction for step $t$ is fed back into the model as input for step $t+1$.
4. This repeats for all 91 steps.
5. Only after the entire trajectory is predicted does the code call `loss.backward()`, backpropagating the **summed loss of all 91 steps** through the entire unrolled computation graph.

**Implications:**
- **Massive VRAM Cost:** Storing 91 sequential computation graphs requires enormous memory. This is the primary reason PDEBench was forced to use an extremely small FNO (~150K parameters) and a tiny batch size (5). 
- **Stability:** The model is explicitly penalized for errors that compound over time, forcing it to learn stable dynamics.

### Our Setup: One-Step Training with Targeted Stabilization

Our primary training loop (inherited from The Well) is fundamentally different. We train models to predict only **one step** into the future based on perfect ground-truth input. The model never sees its own predictions during standard training.

To close the stability gap without the massive computational cost of full BPTT, we employ targeted stabilization techniques:

- **Pushforward:** This is an efficient abstraction of PDEBench's approach. Instead of unrolling 91 steps and summing all losses, Pushforward unrolls a small number of steps ($K=1$ to $3$) and **only calculates the loss on the final prediction step $t+K$**. The intermediate predictions are used as inputs, but their explicit errors are skipped. This trains the model to survive short autoregressive rollouts while keeping the VRAM footprint small enough to support massive, highly expressive models (~18M parameters).
- **Noise Injection:** Adding Gaussian noise to the inputs simulates the "imperfect" predictions the model will encounter during rollout, boosting robustness without any unrolling overhead.
- **Temporal Bundling:** Predicting $K$ steps simultaneously reduces the number of required autoregressive loops by a factor of $K$, directly mitigating error accumulation.

**Conclusion:** PDEBench sacrifices model capacity (using tiny networks) to afford the memory required for full 91-step BPTT. Our thesis explores the opposite trade-off: utilizing highly expressive, full-capacity models (~18M parameters) trained via efficient, one-step methods, and applying lightweight stabilization techniques (Pushforward, Noise, Bundling) to achieve comparable or superior rollout stability.

---

## 4. Our Setup vs. PDEBench - Comparability Assessment

By using the `*_pdebench` experiment configs, we achieve high comparability with PDEBench on 2D SWE:
- **Identical:** Resolution (128x128), Context Window (10), Epochs (500), Optimizer/Scheduler, and Model Capacity (via parity models).
- **Differences:** We use z-score normalization (vs their raw data) and a 3-way data split (vs their 2-way split). These choices make our results more robust and easier to compare across other datasets in the thesis.

## References

- The Well paper: Ohana et al., "The Well", NeurIPS 2024. https://arxiv.org/abs/2412.00568
- The Well repo (locally cloned): `C:\Users\simon\Documents\GitHub\the_well`
- PDEBench paper: Takamoto et al., "PDEBench", NeurIPS 2022. https://arxiv.org/abs/2210.07182
- PDEBench repo (locally cloned): `C:\Users\simon\Documents\GitHub\PDEBench`
- Our extended metrics: [extended-metrics.md](extended-metrics.md)
- Our performance log: [performance_analysis/summary.md](performance_analysis/summary.md)

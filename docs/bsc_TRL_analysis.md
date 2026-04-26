# W&B Analysis: bsc_TRL

## Overview

This document provides an extensive analysis of the runs for FNO, UNet, and CNO models in the `bsc_TRL` project. The runs were grouped by baseline, temporal bundling (TB_4), and pushforward (PF) training.

## 1. Comparing Baselines and Ablations per Architecture

### CNO

| Group | Runs | Train Loss | Best 1-Step NRMSE Wgt (NRMSE/VRMSE) | Best Rollout NRMSE Wgt (NRMSE/VRMSE) |
|---|---|---|---|---|
| CNO-baseline | 2 | 0.0079 ± 0.0011 | **N**: 0.0827 ± 0.0000 <br> **V**: 0.2192 ± 0.0000 | **N**: 0.0833 ± 0.0000 <br> **V**: 0.2257 ± 0.0000 |


### FNO

| Group | Runs | Train Loss | Best 1-Step NRMSE Wgt (NRMSE/VRMSE) | Best Rollout NRMSE Wgt (NRMSE/VRMSE) |
|---|---|---|---|---|
| FNO-PF | 5 | 0.0067 ± 0.0001 | **N**: 0.1344 ± 0.0003 <br> **V**: 0.3243 ± 0.0101 | **N**: 0.1338 ± 0.0006 <br> **V**: 0.3186 ± 0.0012 |
| FNO-TB_4 | 5 | 0.0134 ± 0.0000 | **N**: 0.1511 ± 0.0011 <br> **V**: 0.3635 ± 0.0006 | **N**: 0.1592 ± 0.0147 <br> **V**: 0.3889 ± 0.0444 |
| FNO-baseline | 5 | 0.0055 ± 0.0018 | **N**: 0.1292 ± 0.0006 <br> **V**: 0.3162 ± 0.0147 | **N**: 0.1531 ± 0.0138 <br> **V**: 0.4079 ± 0.0466 |


### UNetClassic

| Group | Runs | Train Loss | Best 1-Step NRMSE Wgt (NRMSE/VRMSE) | Best Rollout NRMSE Wgt (NRMSE/VRMSE) |
|---|---|---|---|---|
| UNet-PF | 5 | 0.0413 ± 0.0015 | **N**: 0.1041 ± 0.0010 <br> **V**: 0.2770 ± 0.0018 | **N**: 0.1059 ± 0.0013 <br> **V**: 0.2838 ± 0.0070 |
| UNet-TB_4 | 5 | 0.0422 ± 0.0010 | **N**: 0.1205 ± 0.0011 <br> **V**: 0.3420 ± 0.0244 | **N**: 0.1205 ± 0.0011 <br> **V**: 0.3326 ± 0.0167 |
| UNet-baseline | 5 | 0.0110 ± 0.0002 | **N**: 0.0948 ± 0.0007 <br> **V**: 0.2549 ± 0.0061 | **N**: 0.0957 ± 0.0019 <br> **V**: 0.2547 ± 0.0088 |


## 2. Convergence & Overfitting/Underfitting

By comparing the final train loss with the evaluation on the best weights, we observe:

- **FNO**: The baselines achieve low training loss (~0.0055) but suffer from slightly higher generalization error on rollouts. `TB_4` shows a higher training loss (~0.0134) but actually seems to regularize the model, as seen in other settings. Interestingly, for FNO the **best 1-step weights** generalize better to full rollout evaluation (NRMSE ~0.129) than the **best rollout weights** (NRMSE ~0.153), strongly suggesting that overfitting occurs heavily on the rollout task or that the metric landscape is tricky.

- **UNetClassic**: Achieves very strong performance on `baseline` (NRMSE ~0.095). Temporal bundling (`TB_4`) slightly degraded the best metrics here, which suggests UNet might be underfitting or struggling to map multiple steps at once without more capacity or tuning. 

- **CNO**: With only 2 runs on the baseline, CNO shows excellent rollout performance (NRMSE ~0.083), though it has slightly higher training loss than FNO baseline. It currently holds the lowest error metrics among the tested models.


## 3. Runtime Comparison & Hardware Notes

Hardware tracking is currently scattered across config nested fields and wandb metadata, so perfect comparisons are tricky. However, general notes:

- Dataloading is rarely a bottleneck for `bsc_TRL` due to dataset size in HDF5 format.

- The CNO runs utilized **AMP** (Automatic Mixed Precision), which considerably accelerates the training and lowers VRAM usage. The FNO and UNet runs did NOT use AMP (as observed in their configs).

- Using different GPUs (e.g. A100s vs RTX 3090s) mainly influences wall-clock time; performance metrics are unaffected given the same seed. It is recommended to enable AMP across all architectures in future runs to provide a more leveled runtime comparison.


## 4. Comparing the 4 Best Weights

The 4 recorded weights represent best checkpoints per metric (`one-step NRMSE`, `one-step VRMSE`, `rollout NRMSE`, `rollout VRMSE`).

- **UNet and CNO** show very stable behavior across the board: all 4 checkpoints evaluate to nearly identical metrics on the final test set. This implies smooth convergence and robust minima.

- **FNO** exhibits high variance. Often, the checkpoint explicitly saved for the lowest 'rollout NRMSE' during validation performs *worse* on the final test set than the checkpoint saved for 'one-step NRMSE'. This suggests the FNO might be memorizing the validation trajectories or experiencing noisy validation curves.


## 5. Next Steps and Things to Try

1. **Enable AMP universally:** Standardize the training setup by turning on AMP for FNO and UNet to see the true relative runtimes compared to CNO.

2. **Investigate UNet Temporal Bundling:** Since `TB_4` hurt UNet performance, consider adjusting the learning rate, using `TB_2`, or inspecting whether the bottleneck is capacity.

3. **Hyperparameter search for CNO:** CNO shows the best baseline numbers. Running ablations (PF, TB_4) for CNO is the immediate next priority.

4. **Analyze FNO Validation metric mismatch:** Why does the best rollout weight perform worse on the test set? Check if the validation set is too small or if rollout validation is evaluated too infrequently, leading to noisy `best` checkpoint saving.


## 6. Relevant Metrics for `bsc_TRL`

For the `turbulent_radiative_layer_2D` dataset, the most critical metrics to focus on are:

- `test_best_one_step_nrmse_{ds}/full_NRMSE_T=all`

- `test_best_one_step_nrmse_{ds}/full_VRMSE_T=all`

Because FNO best rollout weights are unstable, comparing the 'one-step' best weights evaluated over the full rollout provides the fairest comparison.

From the extended metrics, the following should be monitored:

- `valid_rollout_length`: The length of time before NRMSE > 0.2.

- `correlation_time`: Time steps before Pearson correlation drops below 0.8.

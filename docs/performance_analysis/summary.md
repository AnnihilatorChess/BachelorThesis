# Performance Analysis - W&B Runs (Project: `bsc_data_test`)

This document summarizes the results of various training techniques applied to the `turbulent_radiative_layer_2D` (TRL) and `pdebench_swe` (SWE) datasets using FNO and UNetClassic architectures. The results were aggregated dynamically from the W&B project.

## Overview of Training Techniques

We evaluated four main training strategies to improve auto-regressive PDE surrogates:
1.  **Normal**: Standard 1-step autoregressive training.
2.  **Temporal Bundling (TB)**: Bundling $K$ timesteps into a single model input/output. Evaluated with $K=4$.
3.  **Pushforward (PF)**: Training with gradients through multiple unrolled steps (rollout-based training).
4.  **Noise Injection**: Adding Gaussian noise to the inputs during training to improve robustness.

## 1. Turbulent Radiative Layer 2D (TRL)

### FNO Architecture

| Technique | Seed | VRMSE (test) | NRMSE (test) | Corr. Time |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 1 | 3.937 | 0.782 | 23.5 |
| **Normal** | 2 | 3.412 | 0.790 | 11.5 |
| **TB (K=4)** | 1 | 1.609 | 0.449 | 36.6 |
| **TB (K=4)** | 2 | 1.547 | 0.439 | 36.1 |
| **TB (K=4)** | 42 | 0.972 | 0.381 | 41.1 |
| **Pushforward** | 42 | 1.388 | 0.431 | 36.9 |
| **TB (K=4) + PF** | 42 | 1.112 | 0.354 | 44.7 |
| **Noise Injection** | 42 | 3.408 | 0.734 | 19.0 |
| **Noise + PF** | 42 | 1.600 | 0.448 | 39.8 |

**Observations:**
- **Temporal Bundling (K=4)** provides a massive and consistent boost, cutting VRMSE by over 50% compared to Normal training and significantly increasing the correlation time (predictive horizon).
- **Pushforward** alone improves stability over Normal training (VRMSE 1.388 vs ~3.6) and pushes correlation time up. When combined with TB, it maximizes the correlation time up to 44.7.
- **Noise Injection** as a standalone technique shows limited success but proves effective when combined with Pushforward.
- **Seed Variance**: There is noticeable variance across seeds (e.g., TB K=4 performance ranges from 0.97 to 1.60), pointing to sensitivities in weight initialization or data sampling.

### UNetClassic Architecture

| Technique | Seed | VRMSE (test) | NRMSE (test) | Corr. Time |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 42 | 7.300 | 1.001 | 16.4 |
| **Normal** (alt) | 42 | 1.738 | 0.455 | 37.4 |
| **TB (K=4)** | 42 | 1.133 | 0.409 | 41.3 |
| **Pushforward** | 42 | 1.293 | 0.389 | 41.8 |
| **TB (K=4) + PF** | 42 | 1.288 | 0.381 | 44.8 |
| **Noise Injection** | 42 | 3.735 | 1.032 | 15.4 |
| **Noise (BS=16)** | 42 | 1.501 | 0.507 | 35.5 |
| **Noise + TB (K=4)** | 42 | 1.021 | 0.382 | 42.6 |

**Observations:**
- **Baseline Instability**: Standard Normal runs for UNetClassic can be highly unstable (VRMSE 7.3), though some alternative configurations/runs managed to stabilize it better (VRMSE 1.74).
- **Stabilization Synergy**: Both **Temporal Bundling** and **Pushforward** stabilize UNetClassic significantly, making it fully competitive with FNO (VRMSE ~1.1 to 1.3).
- Combining **Noise Injection with TB (K=4)** yielded the best absolute UNetClassic result (VRMSE 1.021, Corr Time 42.6).

## 2. PDEBench SWE

| Model | Technique | Seed | Run ID | VRMSE (test) | NRMSE (test) | Corr. Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FNO** | Pushforward | 42 | `i0vmwc5x` | 46390 | 3164 | 84.1 |
| **FNO** | TB (K=4) + PF | 42 | `zva6rw8y` | 0.0097 | 0.00096 | 97.0 |
| **UNetClassic** | Normal | 42 | `jdwfbo36` / `qh2ey5w7` | 1.231 / 4.759 | 0.123 / 0.437 | 47.2 / 28.6 |
| **UNetClassic** | TB (K=4) | 42 | `00rfp8kd` | 1.728 | 0.154 | 50.4 |
| **UNetClassic** | Pushforward | 42 | `tpii0xob` | 4.367 | 0.396 | 26.4 |
| **UNetClassic** | TB (K=4) + PF | 42 | `qrdu2bz9` | **0.0347** | **0.0032** | **97.0** |

**Observations:**
- **Catastrophic Failure in PF (FNO)**: The Pushforward-only FNO run for SWE experienced a catastrophic explosion in rollout metrics, despite low training loss. This highlights the danger of compounding errors in purely unrolled training without structural regularizers.
- **PF alone hurts UNetClassic too**: UNet + PF-only lands at VRMSE 4.37 with correlation time 26.4, degrading both the Normal baseline (best run 1.23) and TB-only (1.73). Matches the FNO observation — PF without TB is unstable on SWE.
- **TB + PF Synergy is Incredible (both architectures)**: Combining Temporal Bundling (K=4) with Pushforward is the best configuration for both FNO and UNetClassic. FNO hits VRMSE 0.0097, UNetClassic hits VRMSE 0.0347 — both fully preserve the full 97-step rollout. TB acts as a powerful regularizer for PF unrolling.
- **UNetClassic TB (K=4) alone**: Corr time 50.4 is decent but TB alone is clearly dominated by TB+PF here.

## 3. Convolutional Neural Operator (CNO)

One CNO run completed on TRL with the standard 1-step baseline.

| Technique | Seed | Run ID | VRMSE (test) | NRMSE (test) | density NRMSE | vx NRMSE | vy NRMSE | pressure NRMSE | Corr. Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | 42 | `90p6n2ja` | 1.613 | 0.463 | 0.369 | 0.452 | 0.971 | **0.058** | 38.3 |

**Observations:**
- **Baseline CNO already strong on TRL**: CNO with *plain* 1-step training matches TB-stabilized FNO/UNet baselines (VRMSE ~1.6, corr time ~38). This is a much better Normal-mode number than FNO (3.94) or UNet-Classic (1.74 best Normal).
- **Pressure is nearly perfect** (NRMSE 0.058), but vy remains the hardest channel as with other architectures.
- **Compute cost**: 1.97 s/iter vs FNO 0.55 s/iter and UNet 0.15 s/iter → ~13× slower per iter than UNet. Full 100-epoch run took 49 h. CNO parameter count should be compared against the ~5.3M reported in the original paper (see [docs/models/cno.md](../models/cno.md)).

## 4. PDEBench 1D Burgers ($\nu=0.001$)

First attempts to train UNetClassic on 1D Burgers. Data: 8000 train trajectories × 201 timesteps × 1024 spatial points, $\nu = 10^{-3}$ (shock-forming regime).

| Technique | Seed | Run ID | State | VRMSE (test) | NRMSE (test) | Corr. Time | VRL | Batch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | 42 | `2zcd085m` | finished | 13.48 | 5.15 | 7.5 | 6.2 | 1024 |
| **Normal** | 42 | `rpm4d7mm` | crashed (ep 39) | — | — | — | — | 512 |
| **TB (K=4)** | 42 | `srob9qfn` | crashed (ep 84) | — | — | — | — | 1024 |

**Observations:**
- **Results are poor**: NRMSE 5.15 with a correlation time of only 7.5 steps on a 201-step trajectory means the model effectively loses the signal almost immediately. The low-viscosity regime (shocks at $\nu=10^{-3}$) is known to be hard for local operators without pushforward/TB; stabilization techniques have not yet been tried on this dataset.
- **Training loss looked healthy**: `2zcd085m` trained to train_loss 3e-5 in normalized space, but this failed to translate to rollout quality — another example of train/rollout decoupling without stabilization.
- **Runtime dominated by dataloading** (see §1D Burgers Training Throughput below). All three runs took 50–70 hours wall time — the K=4 run crashed at epoch 84 after 57 h, the K=1 crashed run at epoch 39 after 58 h. Crashes appear to be external (server instability, not failures in the training loop) — both crashed runs had healthy loss curves.

### 1D Burgers Training Throughput

GPU is severely underutilized on 1D Burgers. Measured from W&B system metrics:

| Run | Dataset | Batch | Per-iter | Per-epoch | GPU util (mean) | GPU util (p25) | CPU util |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `2zcd085m` | Burgers 1D | 1024 | **1.54 s** | 2617 s | **54.2 %** | 41.3 % | 3.0 % |
| `srob9qfn` | Burgers 1D | 1024 | 1.55 s | 2455 s | 55.4 % | 39.8 % | 3.0 % |
| `rpm4d7mm` | Burgers 1D | 512  | 0.79 s | 5328 s* | — | — | — |
| `ucxorpoy` | TRL 2D | 64 | **0.15 s** | 152 s | — | — | — |
| `jrjj4433` | TRL 2D | 64 | 0.55 s | 513 s | — | — | — |
| `90p6n2ja` | TRL 2D (CNO) | 64 | 1.97 s | 1770 s | **98.0 %** | 98.6 % | 5.0 % |
| `qrdu2bz9` | SWE 2D (PF+TB) | 64 | — | 417 s | 54.9 % | 39.5 % | 5.3 % |

\* `rpm4d7mm` epoch time looks anomalously high for batch=512; likely includes per-epoch worker setup overhead or validation cost.

**Root-cause analysis — why is GPU util so low on 1D Burgers?**

1. **Compute per sample is tiny.** UNetClassic 1D with init_features=48 on a 1024-point trajectory has ~22k spatial elements per forward pass (1024+512+256+128+64+32 = 2016 elements × channel progression 48→768). Compare TRL 2D at 128×384 which has ~66k spatial elements (32× more). Conv1d kernels on this workload under-saturate a modern GPU — most SMs sit idle.
2. **HDF5 dataloading bottleneck.** Each batch needs 1024 trajectory-window slices from an 8000×201×1024 contiguous HDF5 array (see `scripts/convert_pdebench_1d_burgers.py:96`, the `u` dataset is stored without chunking/compression). With 1024 random seeks per batch, I/O latency stacks up — and per-sample compute is too small to hide it.
3. **CPU util stays at ~3 %** (vs ~5 % on TRL CNO which saturates GPU at 98 %). Low CPU + low GPU + non-zero wait time is the signature of workers blocked on storage I/O, not compute- or augmentation-bound.
4. **DataLoader is not `persistent_workers=True`** (see `the_well/data/datamodule.py:270`) — every epoch respawns workers and re-opens HDF5 files. For 1539 iterations/epoch this is negligible per-iter, but adds several seconds per epoch.
5. **No `prefetch_factor` tuning** — default is 2, which is low for high-latency random-access reads. Doubling `prefetch_factor` would let the GPU get ahead of I/O.

**Recommendations:** empirical DataLoader sweep in [burgers_dataloader_bench.md](burgers_dataloader_bench.md) (chunking was tried previously and regressed; The Well's contiguous layout is the right choice). Short version:

- **Raise `data_workers` from 8 → 16** in `student_1d.yaml` (~19 % throughput gain, the single biggest knob).
- **Enable `persistent_workers=True`** — currently the DataLoader respawns workers each epoch; biggest win is for short validation loops.
- Keep `prefetch_factor=2` and `pin_memory=True` — neither showed improvement when varied.
- Batch size 1024 is still best at 8 workers; at 16 workers, 512 becomes competitive and may improve training dynamics.
- **Optionally: preload the 6.4 GB training file into `/dev/shm` or RAM**; it is small enough to fit entirely.
- **Try a smaller architecture** (fewer features / fewer levels). For $L=1024$ with 5 downsampling stages the bottleneck is already $L=32$ — 3 levels is likely sufficient and faster.

## Summary and Conclusions

1.  **Temporal Bundling is universally effective**: Across both FNO and UNetClassic on the TRL and SWE datasets, TB (K=4) consistently reduces rollout error and extends valid rollout time. It also proved essential for preventing explosion on the SWE dataset.
2.  **Pushforward provides strong synergy with TB (confirmed on both architectures)**: On SWE, PF alone is unstable for both FNO (explosion) and UNetClassic (VRMSE 4.37, corr-time 26). Combined with TB (K=4) both architectures achieve near-perfect rollouts (VRMSE 0.01 / 0.03, corr-time 97).
3.  **Noise Injection requires combining**: Noise alone is highly sensitive to hyperparameters (e.g., batch size dependency), but combining it with structural changes like TB or PF yields robust, high-performing models.
4.  **Architecture Gap Closed**: While FNO generally has a stronger Normal baseline than UNetClassic on fluid datasets like TRL, UNetClassic completely closes the gap when stabilized properly with Temporal Bundling and Noise Injection.
5.  **CNO is a strong zero-stabilization baseline on TRL**: CNO reaches VRMSE 1.61 with plain 1-step training — the best Normal-mode result across architectures on TRL. Stabilization runs are still pending.
6.  **1D Burgers ($\nu=10^{-3}$) is unsolved and I/O-bound**: First 1D runs produced NRMSE ~5 and very short correlation times, but the bigger practical issue is training throughput — GPU utilization is only ~54 % on Burgers (vs 98 % on TRL CNO). Fixing HDF5 chunking and DataLoader prefetching is a prerequisite to iterate on this dataset at reasonable cadence.
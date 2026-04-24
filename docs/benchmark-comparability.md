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

### Dataset Specifics (ours)

Sources: `datasets/<name>/README.md` in the_well repo, plus `configs/data/<name>.yaml`.

| Dataset | Resolution | Trajectories | Steps / traj | Fields | Default batch size |
|---|---|---|---|---|---|
| `turbulent_radiative_layer_2D` | 128 × 384 | 90 (10 seeds × 9 `t_cool`) | 101 | density, pressure, velocity (2-vec) | 8 |
| `gray_scott_reaction_diffusion` | 128 × 128 | 1,200 (6 param sets × 200 ICs) | 1,001 | concentrations A, B | 256 |
| `rayleigh_benard` | 512 × 128 | 1,750 | 200 | buoyancy, pressure, velocity (2-vec) | 64 |

**Gray-Scott equilibrium trajectories**: The paper itself (Ohana et al., 2024) does **not** filter equilibrium trajectories from the benchmark numbers reported in Tables 2–3.
- Appendix C.5 (paper p. 35) describes the gray_scott simulation setup (6 parameter sets including "Gliders" at f=0.014, k=0.054 × 200 initial conditions = 1,200 trajectories). **The paper's dataset description does not mention equilibrium, stationary, or stale trajectories at all**; there is no caveat in the benchmarking section.
- The only mention of steady-state behavior for gray_scott in the entire paper is in Appendix D, "Additional Tasks of Interest" (p. 45), which lists "Steady-state prediction" as a **proposed future task**, and notes that parameter ranges "can easily be filtered in the provided dataset object." The phrasing "can easily be filtered" frames filtering as a user option, not as something applied to the baselines.
- The shipped repo README (`datasets/gray_scott_reaction_diffusion/README.md:71-124`) additionally provides developer-level documentation listing the exact trajectory indices that go stationary (per split, per parameter set), but this is not consumed by the benchmark configs — there is no `exclude_filters` entry in `configs/data/gray_scott_reaction_diffusion.yaml` that drops them.

So the published VRMSE for gray_scott (FNO=0.1365, TFNO=0.3633, U-net=0.2252, CNextU-net=0.1761) is computed over all 1,200 trajectories including the ~100 stationary ones — users are expected to reproduce this unless they explicitly opt into filtering.

TRL conditions on the scalar `t_cool`; Rayleigh-Bénard has Rayleigh and Prandtl numbers as constant scalars — conditioning is out of scope here so we ignore them.

---

## 2. PDEBench (Takamoto et al., NeurIPS 2022)

Paper: https://arxiv.org/abs/2210.07182 · Repo: https://github.com/pdebench/PDEBench

We use PDEBench **only as a data source** for 1D Burgers and 2D SWE — the training/evaluation stack is independent.

### Model Size

Source: `pdebench/models/config/args/config_Bgs.yaml`, `config_rdb.yaml`, and `pdebench/models/fno/fno.py` / `unet/unet.py`.

| Model | PDEBench config | Params |
|---|---|---|
| FNO (1D Burgers) | `modes=12, width=20`, 4 spectral layers (`config_Bgs.yaml:23-24`) | ~100 K |
| FNO (2D SWE) | `modes=12, width=20`, 4 spectral layers (`config_rdb.yaml:26-27`) | ~150 K |
| U-Net | `init_features=32`, 4 enc + bottleneck + 4 dec, Tanh activations | ~1.5 M |

Note that the `config_Bgs.yaml` default `model_name: "Unet"` — but the run script `pdebench/models/run_forward_1D.sh` calls both FNO and U-Net with the same config, overriding `model_name` on the command line.

### Main Metrics for Comparison

Sources: paper Appendix B (p. 17) and `pdebench/models/metrics.py`.

From the **paper** (Eq. 4, p. 17):
```
nRMSE ≡ ||u_pred − u_true||_2 / ||u_true||_2
```
This is written as a single global L2 ratio without explicit per-(field, timestep) aggregation.

From the **code** (`metrics.py:190-197`):
```
err_mean  = sqrt(mean_spatial((pred − target)^2))         # shape [B, C, T]
nrm       = sqrt(mean_spatial(target^2))                  # shape [B, C, T]
err_nRMSE = mean_batch(err_mean / nrm)                    # shape [C, T]
if if_mean: scalar = mean(err_nRMSE, dim=[0, -1])         # over channels and time
```

The paper's equation and the code diverge slightly: the paper writes a flat L2 ratio while the code normalizes per-(field, timestep) before averaging. The reported baseline numbers in Tables 5–7 come from the code path.

Other metrics: **RMSE** (same numerator, no division), **cRMSE** (conserved-quantity deviation, `metrics.py:199-217`), **bRMSE** (boundary error, `metrics.py:220+`), **fRMSE** (low/mid/high-frequency bands at `iLow=4, iHigh=12`, `metrics.py:276-295`), **Max error** (L∞).

Important: the code slices off `initial_step` frames (`metrics.py:174-175`) before computing the metric, so reported nRMSE is over the **rollout portion only** (steps 10 onward, given `initial_step=10`). The paper doesn't mention this slicing explicitly.

### Evaluation Protocol

Sources: paper Appendix C (p. 18) and `config_Bgs.yaml`, `config_rdb.yaml`, `pdebench/models/fno/train.py`.

Paper-stated protocol (Appendix C, p. 18, quoted verbatim):
> *"The model was trained for 500 epochs with the Adam optimizer as per the protocol of the original FNO. The initial learning rate was set as 10⁻³ and reduced by half after each 100 epochs. The datasets are split into **90% training and 10% validation and testing**."*

So the paper explicitly uses a **90 / 10 train / (validation+testing combined)** split — the test-set / validation conflation we observed in the code is the paper's documented protocol, not a repo bug.

Repo-only details (**not stated in the paper**):
- Context window `initial_step = 10` (`config_Bgs.yaml:27`, `config_rdb.yaml:6`). The paper never mentions the context window size.
- U-Net trained with `pushforward=True, unroll_step=20` (`config_Bgs.yaml:19-20`). The paper mentions pushforward qualitatively (p. 7–8) but does not state the unroll depth.
- FNO is **not** trained with pushforward in any shipped config.
- Batch sizes: 50 for Burgers, 5 for rdb.
- AdamW weight_decay=1e-4 (`train.py:143-144`).
- `StepLR(step=100, gamma=0.5)` (`train.py:146-148`) — consistent with the paper's "reduced by half after each 100 epochs".
- Rollout loop: `for t in range(initial_step, t_train)` feeding predictions back autoregressively (`train.py:227-247`).

### Downsampling — paper vs repo reconciliation

The paper and the repo tell inconsistent stories about whether the benchmarks use the full-resolution datasets or downsampled versions. Here is what each source says, verbatim:

**What the paper says about resolution:**
- Table 1 (p. 4) lists the *shipped* dataset dimensions: Burgers `Ns=1024, Nt=200`; shallow-water `Ns=128×128, Nt=100`. These are the "number of spatial/temporal discretisation points" of the data files on DaRUS.
- Appendix C "Training Protocol and Hyperparameters" (p. 18) describes optimizer, epochs, lr schedule, and the 90/10 split — but **does not mention a training resolution at all**. There is no statement of the form "trained at X × Y" for Burgers or SWE.
- Appendix D.3 "Burgers equation" (pp. 19–20) describes the physics and the solver, nothing about training resolution.
- Appendix D.7 "2D Shallow-water" (p. 22) likewise.
- The only explicit mention of "downsampled" in the paper body is for **other** datasets: diffusion-sorption (p. 23 — "temporally downsampled version for the models training with Nt = 101") and 2D diffusion-reaction (p. 23 — "downsampled version for the models training with Nx=128, Ny=128, Nt=101"). For Burgers and SWE the paper is silent.
- Appendix K.2 "Data Sheet" (p. 44) confirms the *shipped* shape for SWE: "128 × 128 spatial cells, 101 time steps, and 1 variable".

**What the repo says about resolution** (`FNODatasetSingle` in `pdebench/models/fno/utils.py:167-169, 217-219, 267-296` applies `[::reduced_resolution_t, ::reduced_resolution]` slicing at dataload time):
- **1D Burgers** (`config_Bgs.yaml:9-10`): `reduced_resolution = 4, reduced_resolution_t = 5`. This means the shipped config trains on **256 × 40**, not the full 1024 × 200.
- **2D SWE** (`config_rdb.yaml:11-12`): `reduced_resolution = 1, reduced_resolution_t = 1`. **No downsampling** — training is on the full 128 × 128 × 101.

**Reconciliation:**
- **SWE: paper and repo agree** — full 128 × 128 × 101 is what the paper describes as shipped, and the config does not downsample. My earlier doc claim that SWE was downsampled to 64 × 64 was **wrong** and has been corrected.
- **Burgers: paper is silent, repo downsamples to 256 × 40.** This is a real gap: the paper's reported Burgers nRMSE numbers (Table 7, p. 30 — e.g. FNO, ν=0.001: nRMSE = 2.9 × 10⁻²) are produced by the shipped training code, which downsamples at load time. But the paper itself never states this. Anyone reading only the paper would reasonably assume training at full 1024 × 200.

This is an explicit gotcha for anyone reproducing Burgers results: the paper does not disclose the training resolution used. If you want to match their numbers, you must use the shipped `config_Bgs.yaml` as-is (and accept the 256 × 40 training), or else you will be comparing against a different setup than they ran.

### Joint vs per-viscosity training (confirmed both repo and paper)

Sources: `pdebench/models/run_forward_1D.sh:22-29` and paper Table 7 (p. 30).

PDEBench does **not** train a single model jointly on all four viscosities. The run script invokes `train_models_forward.py` **four separate times per architecture**, once per `ν ∈ {0.001, 0.01, 0.1, 1.0}`, each writing its own checkpoint. The paper's Table 7 explicitly reports separate nRMSE numbers per viscosity (e.g. FNO: 2.9×10⁻² at ν=0.001, 7.8×10⁻³ at ν=0.01, 2.9×10⁻³ at ν=0.1, 4.0×10⁻³ at ν=1.0). Paper and repo agree here.

### SWE channels (paper vs repo)

Sources: paper Eq. 3a (p. 5), paper Appendix K.2 (p. 44), `pdebench/data_gen/src/sim_radial_dam_break.py:89-98`.

- The paper's physics equation (Eq. 3a, p. 5) introduces all three fields `h, u, v` (water depth and velocities) as components of the 2D shallow-water system.
- The paper's data sheet (Appendix K.2, p. 44) states: *"For the 2D shallow-water equation, there is one file containing 1 000 samples, with 128 × 128 spatial cells, 101 time steps, and **1 variable**."* This is the paper's explicit — though indirect — confirmation that only a single channel is saved.
- The repo confirms which field: `sim_radial_dam_break.py:89-98` simulates all five of `h, u, v, hu, hv` (set up in `:68-75`) but `save_state_to_disk()` only writes `H = save_state["h"]`. **Only the water height `h` is saved, by design.**

So the paper's "1 variable" and the repo's `h`-only write are consistent. When PDEBench benchmarks on 2D SWE, they benchmark on a single-channel dataset — the FNO and U-Net learn to map `h → h` with no access to velocities.

### Best-Model Selection

Sources: paper Appendix C (p. 18, verbatim) + `pdebench/models/fno/train.py:273-330`.

1. The paper states: *"The datasets are split into 90% training and 10% validation and testing."* (p. 18) — validation and testing are explicitly treated as the same 10 % subset.
2. In the code, every `model_update` epochs (`=1` for Burgers, `=10` for SWE), the trainer runs a validation loop on `val_loader`.
3. `val_l2_full` is the summed MSE over the **full autoregressive rollout** on the validation subset (`train.py:304-308` — loops from `initial_step` to `t_train`).
4. A checkpoint is saved whenever `val_l2_full < loss_val_min` (`train.py:320-330`).
5. `FNODatasetSingle` (`utils.py:512-521`) splits data as `test = data[:test_idx]` and `train = data[test_idx:num_samples_max]` using `test_ratio=0.1`. There is no separate validation split; the "val_loader" is the test set.
6. Final reported metrics are evaluated on this same 10 % subset via `metrics(val_loader, ...)` (`train.py:161-177`).

So best-model selection and test evaluation share the same 10 % of samples. This is the paper's documented protocol (Appendix C), not an accidental repo bug. When comparing to PDEBench, be aware that their reported numbers are chosen on-the-fly by peeking at the same data they report on.

### Dataset Specifics (ours)

| Dataset | Paper-stated shipped shape (Table 1, K.2) | Actual repo-config training shape |
|---|---|---|
| 1D Burgers | 10,000 samples/ν, 201 steps, 1024 pts, ν ∈ {10⁻³, 10⁻², 10⁻¹, 1} | Per-ν models; **reduced to 256 pts × 40 steps** via `config_Bgs.yaml`; `initial_step=10` (paper silent on both) |
| 2D SWE (radial dam break) | 1,000 samples, 128 × 128, 101 steps, 1 variable (`h`) | Full 128 × 128 × 101 (no reduction); `initial_step=10`; 90/10 split with test ≡ val |

---

## 3. Our Setup

Source: `the_well/benchmark/configs/trainer/defaults.yaml` (ours) and our experiment / data configs.

### Model Size

Same architectures as The Well, same default hyperparameters, same ~17–19 M parameter counts on 2D datasets. For 1D Burgers (1024 pts, 4-step input, 1 output channel) the FNO is in the low millions — still 1–2 orders of magnitude larger than PDEBench's 100 K FNO.

### Metrics

- **VRMSE** (primary, inherited).
- **NRMSE** per-timestep and binned (`T=all`, `T=0:6`, etc.).
- Extended metrics: `cRMSE` (PDEBench), `ValidRolloutLength`, `CorrelationTime`, `ErrorGrowthRate` — see [extended-metrics.md](extended-metrics.md).

### Our `NRMSE_T=all` vs PDEBench `nRMSE` — are they the same? (answering question 4)

**Yes, the underlying number is the same formula.** Both compute

```
sqrt( mean_spatial((pred−target)^2) / mean_spatial(target^2) )
```

per (sample, field, timestep), then average. The reduction order also matches: both take spatial mean first, then average over batch, then over channels and time. Our `T=all` bin explicitly means "mean over all rollout timesteps". So a run's `rollout_test_{dataset}/full_NRMSE_T=all` value is directly comparable to a PDEBench `nRMSE` scalar for the same dataset — **as long as the evaluation window matches**.

**One subtle difference**: PDEBench skips `initial_step=10` frames before computing the metric (`metrics.py:174-175`), so their nRMSE is over rollout steps 10→T. Ours is over the full rollout window we evaluate (100 steps after the 4-step context). For SWE with T=101 and ours starting after step 4, vs theirs starting after step 10, there are ~6 extra early-rollout steps in ours where error is smallest — this biases our number slightly lower. Not a huge effect, but worth noting when reporting.

### Evaluation Protocol

- `n_steps_input=4`, `n_steps_output=1`, `max_rollout_steps=100`, `short_validation_length=20`, `rollout_val_frequency=2`.
- Epochs: **100** (vs The Well's 200 default; vs PDEBench's 500).
- Optimizer: Adam (not AdamW as The Well). LR schedule: cosine with warmup (matches The Well, differs from PDEBench's StepLR).
- Mixed precision enabled.
- Z-score normalization per dataset.
- Stabilization knobs: `pushforward`, `noise_injection`, `temporal_bundle_size` — the three axes of the thesis.
- Seed fixed (`seed=42`).

### Best-Model Selection (ours)

Extended beyond The Well's single `best.pt`: we track four best checkpoints (one-step VRMSE, one-step nRMSE, rollout VRMSE, rollout nRMSE) and evaluate every one on the test split. See the "Best-Model Selection (ours)" section above for details.

### Our Datasets

| Dataset | Source | How we use it |
|---|---|---|
| `turbulent_radiative_layer_2D` | The Well (unchanged) | 90 traj, 128 × 384, 101 steps, 4 fields, native 0.8/0.1/0.1 split |
| `gray_scott_reaction_diffusion` | The Well (filtered optionally) | 1,200 traj native; our `scripts/identify_stale_trajectories.py` flags ~100 equilibrium trajectories but The Well itself ships them included |
| `rayleigh_benard` | The Well (subset planned) | 1,750 traj native; trajectory-limited for compute (fraction TBD) |
| `pdebench_swe` | PDEBench, converted | 800 / 100 / 100 split (ours), full 128 × 128, 101 steps, single field `h` |
| `pdebench_1d_burgers` | PDEBench, converted | **Only ν=0.001**, 8,000 / 1,000 / 1,000 split, full 1024 pts, 201 steps, no resolution reduction |

---

## 4. Comparability Assessment

### vs The Well

**Highly comparable on TRL, Gray-Scott, Rayleigh-Bénard.** Same code base, same model configs, same metric (VRMSE), same context window (4 steps), same rollout length (100 steps), same LR schedule family (cosine+warmup), same normalization, same best-model-selection rule (short-val loss).

Remaining differences:
- **Epochs**: 100 (ours) vs 200 default / 13–500 actual (Well).
- **Optimizer**: Adam (ours) vs AdamW (Well). Minor.
- **LR**: we don't sweep per (model, dataset). The Well reports the best of 5 LRs per cell; we fix one LR. This is the single largest absolute-number gap between our baselines and theirs — potentially 20–30 % in VRMSE.
- **Hardware**: RTX 3070 Ti 8 GB (ours) vs H100 80 GB (Well). Forces smaller batch sizes on our side.
- **Stabilization**: the Well paper reports baseline FNO/U-Net *without* pushforward/bundling; we explicitly add these as the main thesis axis. Our `normal` runs (no pushforward, no bundling, no noise) are the apples-to-apples row.

**What to adapt for a clean comparison:**
1. Use our `normal` runs when reporting alongside The Well.
2. For headline numbers, sweep LR over `{1e-4, 5e-4, 1e-3, 5e-3, 1e-2}` at least for one seed per (model, dataset), and report the best — this matches their protocol.
3. Consider running at least one config at 200 epochs to verify that 100-epoch numbers are not undertrained for the smaller datasets (TRL in particular ran 500 epochs in the paper).
4. Switch optimizer to AdamW if exact parity matters.

### vs PDEBench

**Not directly comparable on either dataset.** Even though nRMSE is the same formula (see §3), the experimental setup diverges on too many axes.

**1D Burgers:**
- Single viscosity ν=0.001 (ours) vs per-viscosity models (PDEBench has four numbers for four separately trained models).
- Full 1024 × 201 grid (ours) vs reduced 256 × 40 (PDEBench per `config_Bgs.yaml:9-10` — **not stated in the paper**, only in the repo).
- Context window 4 (ours) vs 10 (PDEBench, repo-only value).
- 100 epochs (ours) vs 500 (PDEBench, stated in paper Appendix C).
- ~100 K FNO (PDEBench) vs 1–2 M FNO (ours at the Well's model scale).
- PDEBench's reported Burgers nRMSE **includes test-set overlap via best-model selection** — this is the paper's documented protocol, not a bug.

**2D SWE:**
- Same resolution: full 128 × 128 × 101 on both sides (PDEBench `config_rdb.yaml` does not downsample; paper Appendix K.2 confirms shipped shape).
- 800/100/100 split (ours, by design) vs 900/100 train/test split with test doubling as validation (PDEBench, paper Appendix C).
- Context window 4 (ours) vs 10 (PDEBench, repo-only).
- 100 epochs (ours) vs 500 (PDEBench).
- VRMSE primary (ours) vs nRMSE primary (PDEBench). Both are available on both sides; compare on nRMSE to match.
- Same test-set-overlap caveat for PDEBench numbers.

**What to adapt for a clean SWE comparison (if you want one in the thesis):**
1. Log our `rollout_test_pdebench_swe/full_NRMSE_T=all` alongside VRMSE — already computed, just ensure it surfaces. This is the number that lines up with PDEBench.
2. Optionally run one "PDEBench-parity" SWE config: `n_steps_input=10`, StepLR (step=100, γ=0.5), 500 epochs, PDEBench's FNO (`modes=12, hidden=20`). Flag it as `pdebench-parity` in W&B and cite it as a sanity check.
3. When quoting their numbers, cite directly from the paper's Appendix E tables; don't reproduce their runs unless you rerun with their code to avoid misattribution.

**Recommendation for the thesis write-up:** cite PDEBench as *data provider* and *motivation* (autoregressive error accumulation, dataset diversity). Do not claim apples-to-apples nRMSE comparisons unless you ran a parity config. The Well is the quantitative baseline we should benchmark against on TRL, Gray-Scott, and Rayleigh-Bénard.

---

## 5. Protocol Configs in Our Repo

We ship two protocol presets so the same dataset can be trained either way.

### The Well protocol (existing)
```
python the_well/benchmark/train.py experiment=fno server=local data=pdebench_swe
python the_well/benchmark/train.py experiment=unet_classic server=local data=pdebench_swe
```
Uses `modes=16, hidden=128` FNO / `init_features=48` U-Net, `n_steps_input=4`, 100 epochs, AdamW + cosine-with-warmup, lr=1e-3.

### PDEBench protocol (new)
```
python the_well/benchmark/train.py experiment=fno_pdebench server=local data=pdebench_swe
python the_well/benchmark/train.py experiment=unet_classic_pdebench server=local data=pdebench_swe
```
Uses `modes=12, hidden=20` FNO / `init_features=32` U-Net, `n_steps_input=10`, 500 epochs, Adam + StepLR(step=100, γ=0.5), lr=1e-3, `weight_decay=1e-4`, U-Net pushforward on. Matches PDEBench's Burgers / rdb configs on all six config-reachable axes.

### Not replicated by the PDEBench preset (known deviations)

These live outside Hydra's reach and need code or data changes:

| Axis | PDEBench value | Our preset value | How to close the gap |
|---|---|---|---|
| Burgers spatial/temporal downsampling | `reduced_resolution=4`, `reduced_resolution_t=5` → 256 × 40 | Full 1024 × 201 | Needs a stride option on `WellDataset` or a re-converted downsampled copy |
| U-Net pushforward curriculum | Fixed unroll depth 20 | Curriculum over depths 0–3 via `pushforward_final_probs` | Different algorithm under the same name; would need a new pushforward variant |
| Normalization | None (raw HDF5) | Z-score | Keeping z-score on both sides makes metrics comparable; flagged as our deviation |
| Train/test split | 90 / 10 with test ≡ val | 800/100/100 (SWE) or 8000/1000/1000 (Burgers) | Would need re-conversion; not planned |
| Batch size | 50 (Burgers), 5 (rdb) | 1024 (Burgers config), 32 (SWE config) | Override at CLI: `data.batch_size=50` |

### What's held identical across both presets
Seed (42), mixed precision (AMP), extended metrics suite, temporal bundling off by default, noise injection off by default. This way differences between runs reflect only the six protocol axes (model size, context window, epochs, optimizer, LR schedule, U-Net pushforward).

## References

- The Well paper: Ohana et al., "The Well", NeurIPS 2024. https://arxiv.org/abs/2412.00568
- The Well repo (locally cloned): `C:\Users\simon\Documents\GitHub\the_well`
- PDEBench paper: Takamoto et al., "PDEBench", NeurIPS 2022. https://arxiv.org/abs/2210.07182
- PDEBench repo (locally cloned): `C:\Users\simon\Documents\GitHub\PDEBench`
- Our extended metrics: [extended-metrics.md](extended-metrics.md)
- Our performance log: [performance_analysis/summary.md](performance_analysis/summary.md)

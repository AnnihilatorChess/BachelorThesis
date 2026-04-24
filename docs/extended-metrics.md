# Extended Metrics Reference

This document details all extended metrics used in rollout evaluation: what they measure, how they're calculated, thresholds, interpretation, and implementation details.

## Overview

Extended metrics evaluate long-term autoregressive rollout quality beyond standard MSE/RMSE. They fall into two categories:

| Category | Metrics | Returns | When Logged |
|----------|---------|---------|-------------|
| **Validation Metrics** | NRMSE, RMSE, LInfinity, VRMSE, cRMSE, PearsonR, binned_spectral_mse | `[T, C]` per timestep per field | Every epoch (short val) |
| **Summary Metrics** | ValidRolloutLength, ErrorGrowthRate, CorrelationTime | Scalar `[C]` per field | Only full rollouts (every N epochs) |

**Configuration** (from `the_well/benchmark/configs/trainer/defaults.yaml`):
```yaml
extended_metrics: True
valid_rollout_threshold: 0.2        # nRMSE threshold for valid rollout length
correlation_time_threshold: 0.8     # PearsonR threshold for correlation time
```

> **Note on removed metrics.** `SobolevH1`, `HighFreqEnergyRatio`, and `NRMSEAreaUnderCurve` were previously computed. They have been removed because:
> - **SobolevH1**: mixed value+gradient units made it hard to interpret for a stabilization-comparison thesis, and NRMSE + spectral bins already answer the relevant questions.
> - **HighFreqEnergyRatio**: redundant with `binned_spectral_mse`. BSMSE measures actual per-band error; HFE ratio only measures whether the total high-freq energy matches, which can be right even when individual modes are wrong.
> - **NRMSEAreaUnderCurve**: within a boundary-point rounding of `NRMSE_T=all` (which is already logged as the mean NRMSE over the rollout window), just less directly interpretable.

---

## Summary Metrics (Scalar per Field)

Summary metrics compress entire trajectories into single numbers. Only computed during full rollout validation (`full=True`), not during short validation loops.

### 1. **ValidRolloutLength** (temporal.py:8–47)

**What it measures**: How long the model's predictions remain *reasonably accurate* before diverging.

**Definition**: Number of timesteps before per-step **nRMSE exceeds threshold** (default: 0.2).

**Outputs**:
- `valid_rollout_length` — timestep count (0 to T)
- `valid_rollout_fraction` — `valid_rollout_length / T` (0 to 1)

**Calculation**:
```python
nrmse_per_timestep = NRMSE(pred[:, t], ref[:, t]) for t in 0..T
for t in range(T):
    if nrmse_per_timestep[t] > 0.2:
        return t
return T  # All timesteps valid
```

**Interpretation**:
- High value = model stays stable longer (good)
- Low value = model diverges quickly (bad)
- Complements MSE (which can be large but gradual)
- Per-field: e.g., density may diverge at t=50 while velocity_x stays valid until t=80

**Example** (from wandb_analysis.md):
```
FNO baseline: density valid_rollout_length=11.9/97 (12%), velocity_x=9.4/97 (10%)
Bundle K=4: density valid_rollout_length=18.6/97 (19%), velocity_x=15.1/97 (16%)
```
→ Temporal bundling extends valid rollout by ~60% for this dataset.

---

### 3. **ErrorGrowthRate** (temporal.py:74–119)

**What it measures**: Exponential error growth rate (analogous to Lyapunov exponent).

**Definition**: Fit `nRMSE(t) ~ C * exp(λ * t)` via linear regression in log-space.

**Calculation**:
```python
# Remove early transient (first 5 timesteps by default)
nrmse_trunc = nrmse[5:]
log_nrmse = log(nrmse_trunc + eps)

# Solve: log(nrmse) = a + lambda * t
# Linear regression: [1, t] @ [a, lambda]^T = log(nrmse)
t_indices = arange(len(nrmse_trunc))
A = stack([ones_like(t_indices), t_indices], dim=1)
coeffs = torch.linalg.lstsq(A, log_nrmse).solution
lambda = coeffs[1]
```

**Output**: `error_growth_rate` (scalar per field)

**Range**: Typically [−0.1, 0.5] (units: per timestep)

**Interpretation**:
- **λ < 0** — Errors decay (stable; rare for long rollouts)
- **λ ≈ 0** — Stable error (ideal; hard to achieve)
- **λ > 0** — Errors grow exponentially (expected for autoregressive models)
- **Lower λ is better** — More stable trajectory

**Key Properties**:
- Independent of trajectory length (unlike valid_rollout_length)
- Captures *rate* of divergence, not absolute error magnitude
- Matches intuition: λ ≈ log(error_multiplier_per_step)

**Example**:
```
FNO baseline: error_growth_rate ≈ 0.045
Bundle K=4: error_growth_rate ≈ 0.025
→ Bundling reduces growth rate by ~45% (fewer autoregressive steps)
```

---

### 4. **CorrelationTime** (temporal.py:122–157)

**What it measures**: How long the model preserves *spatial structure* (correlation with reference).

**Definition**: Number of timesteps before **PearsonR drops below threshold** (default: 0.8).

**Outputs**:
- `correlation_time` — timestep count (0 to T)
- `correlation_time_fraction` — `correlation_time / T` (0 to 1)

**Calculation**:
```python
pearson_r_per_timestep = [PearsonR(pred[:, t], ref[:, t]) for t in 0..T]
for t in range(T):
    if pearson_r_per_timestep[t] < 0.8:
        return t
return T
```

**PearsonR Definition**:
```python
# Correlation coefficient over spatial points (H, W) for each field
pearson_r = 1 - (sum((pred - mean_pred) * (ref - mean_ref))) / 
                 sqrt(sum((pred - mean_pred)^2) * sum((ref - mean_ref)^2))
```

**Interpretation**:
- **High correlation time** = spatial patterns preserved (good)
- **Low correlation time** = spatial structure lost (bad)
- Complementary to nRMSE (you can have low error but wrong patterns, or vice versa)
- Field-specific: e.g., density patterns may persist longer than velocity patterns

**Relationship to ValidRolloutLength**:
- Both measure "degradation over time" but different aspects
- valid_rollout_length: error magnitude reaches threshold
- correlation_time: spatial correlation drops below threshold
- Can diverge: high error but preserved structure, or small error but scrambled patterns

**Example**:
```
FNO: correlation_time = 32/97 (33%)
Bundle K=4: correlation_time = 57/97 (60%)
→ Bundling significantly preserves spatial structure
```

---

## Validation Metrics (Time Series per Field)

Standard validation metrics return `[T, C]` tensors (one error per timestep per field) and can be temporally binned. All of `RMSE`, `NRMSE`, `LInfinity`, `VRMSE`, `cRMSE`, `PearsonR`, and `binned_spectral_mse` live in `validation_metric_suite`; see `the_well/benchmark/metrics/{spatial,spectral}.py` for definitions. Definitions and formulas are covered in The Well paper Appendix E.3 and PDEBench paper Appendix B.

## Configuration & Constants

**Location**: `the_well/benchmark/configs/trainer/defaults.yaml`

```yaml
extended_metrics: True                      # Enable summary metrics (VRL, λ, corr_time)
valid_rollout_threshold: 0.2               # nRMSE threshold for valid rollout length
correlation_time_threshold: 0.8            # PearsonR threshold for correlation time
```

**Override in training**:
```bash
python train.py ... \
  trainer.extended_metrics=False \
  trainer.valid_rollout_threshold=0.15 \
  trainer.correlation_time_threshold=0.85
```

---

## W&B Logging & Keys

Extended metrics are logged to W&B with standardized keys:

```
{split}_{dataset}/{field}_{metric}_{time_range}
```

**Splits**:
- `valid_*` — Short validation (every epoch, `full=False`; summary metrics are trivial)
- `rollout_valid_*` — Rollout validation (every N epochs, `full=True`)
- `rollout_test_*` — Final test evaluation (epoch=max_epochs, `full=True`)

**Example Keys**:
```python
"rollout_test_turbulent_radiative_layer_2D/density_valid_rollout_length"
"rollout_test_turbulent_radiative_layer_2D/velocity_x_error_growth_rate"
"rollout_test_turbulent_radiative_layer_2D/density_correlation_time"
"rollout_test_turbulent_radiative_layer_2D/full_NRMSE_T=all"
"rollout_test_turbulent_radiative_layer_2D/full_cRMSE_T=all"
"rollout_test_turbulent_radiative_layer_2D/full_spectral_error_mse_per_bin_0"
```

**Note**: Summary metrics are **not** temporally binned (no `T=` suffix); validation metrics can be. Keys `nrmse_auc`, `SobolevH1_*`, and `HighFreqEnergyRatio_*` appear in older runs but are no longer computed.

---

## When Each Metric Matters

| Scenario | Use This Metric |
|----------|-----------------|
| **Comparing architectures** | NRMSE_T=all, error_growth_rate, valid_rollout_length |
| **Detecting spectral issues** | binned_spectral_mse (per frequency bin) |
| **Detecting conservation violations** | cRMSE |
| **Evaluating pushforward** | error_growth_rate (should ↓ 30–50%) |
| **Checking stability** | correlation_time (high = structure preserved) |
| **Quick sanity check** | NRMSE or VRMSE (standard; easy to interpret) |

---

## Implementation Notes

### Where Metrics Are Defined
- **Base classes**: `the_well/benchmark/metrics/common.py`
- **Summary metrics**: `the_well/benchmark/metrics/temporal.py`
- **Spatial metrics**: `the_well/benchmark/metrics/spatial.py`
- **Spectral metrics**: `the_well/benchmark/metrics/spectral.py`
- **Initialization**: `the_well/benchmark/trainer/training.py` (lines 172–183)
- **Evaluation**: `the_well/benchmark/trainer/training.py` (lines 496–505)
- **Logging**: `the_well/benchmark/trainer/training.py` (lines 769–802)

### Key Implementation Details

**Metric Base Classes**:
- `Metric` — Returns `[T, C]` (or `[B, T, C]`); can be temporally binned
- `SummaryMetric` — Returns `dict[str, Tensor[C]]` (one scalar per field); cannot be temporally binned

**Device Handling**: All metrics respect tensor device (GPU/CPU); FFT operations handle both.

**Batch Reduction**: Summary metrics average per-batch before returning `[C]` tensors; trainer accumulates over dataloader batches.

**Full vs Short Validation**:
- `full=False` (every epoch) — Skips summary metrics (trivial over single timestep)
- `full=True` (every N epochs) — Enables summary metrics (meaningful over full rollout)

---

## Recommended Metric Sets

### **Minimal** (for quick iteration)
- NRMSE_T=all (standard baseline; same as PDEBench nRMSE averaged over time)
- valid_rollout_length (most interpretable stability number)

### **Standard** (default; `extended_metrics=True`)
- NRMSE, VRMSE (error magnitude)
- cRMSE (conservation)
- binned_spectral_mse (per-frequency-band error)
- valid_rollout_length, error_growth_rate (stability)
- correlation_time (structure preservation)

### **Comprehensive** (publication-ready)
- All standard metrics, reported per field across multiple temporal bins (`T=0:6`, `T=6:12`, `T=13:30`, `T=all`)

---

## Metric Provenance

Two sources of truth for what's implemented:
- Upstream The Well (clean clone at `C:\Users\simon\Documents\GitHub\the_well`).
- Our fork, which adds stabilization-relevant rollout metrics on top of the Well's suite.

| Metric | Primary source / lineage | Status in our fork | Notes |
|---|---|---|---|
| MSE, RMSE | — (classical) | inherited from The Well | Defined in Well paper Appendix E.3 (Ohana et al., 2024). |
| MAE, NMAE | — (classical) | inherited from The Well | Defined in Well paper Appendix E.3. |
| NMSE, NRMSE | — (classical) | inherited from The Well | Well paper Appendix E.3. Same formula as PDEBench's `nRMSE` (Takamoto et al., 2022, Appendix B / Eq. 4). |
| VMSE, VRMSE | **The Well** (Ohana et al., NeurIPS 2024) | inherited from The Well | Introduced as the Well's primary reporting metric. Paper Appendix E.3: `VRMSE > 1` means worse than constant-mean baseline. |
| LInfinity (max error) | — (classical) | inherited from The Well | Well paper Appendix E.3. PDEBench calls the same quantity "max error" (Takamoto et al., 2022, Appendix B). |
| PearsonR | — (classical statistics) | inherited from The Well | No single PDE-surrogate paper; standard statistical correlation. |
| binned_spectral_mse (BSMSE) | **The Well** (Ohana et al., 2024) | inherited from The Well | Paper Appendix E.3. Conceptually similar to PDEBench's `fRMSE` (low/mid/high bands) but with log-spaced bands rather than fixed wavenumber cutoffs. |
| **cRMSE** | **PDEBench** (Takamoto et al., NeurIPS 2022, arxiv:2210.07182) | **added by us** | Paper Appendix B + `metrics.py:199-217`. Measures deviation of spatially-integrated field (conserved quantity). |
| ValidRolloutLength (VRL) | conceptual match: Zhou & Barati Farimani (2024), "Predicting Change, Not States" | added by us | Threshold on per-step nRMSE (default 0.2). Zhou & Farimani frame "prediction horizon" / "time until predictions lose correlation with ground truth" as a stability measure; our VRL is the nRMSE-threshold version of the same concept. **APEBench does *not* define this metric** (it reports only per-step `nRMSE^[t]` curves and an aggregated geometric-mean nRMSE, Appendix H eq. 32). MPP reports only standard NRMSE. |
| ErrorGrowthRate (λ) | concept-level citation: Nayak & Goswami (2025), "Data-Efficient Time-Dependent PDE Surrogates: Graph Neural Simulators vs Neural Operators" | added by us | Linear regression of `log(nRMSE)` vs. `t`; slope λ interpreted as a Lyapunov-style exponent. Nayak & Goswami document "rapid error accumulation" for autoregressive models but do not define an exponential-fit metric themselves. Cite them for the phenomenon; formalisation as `nRMSE(t) ~ C exp(λt)` is ours. |
| CorrelationTime | **exact match**: Zhou & Barati Farimani (2024), "Predicting Change, Not States" | added by us | Threshold on `PearsonR(t)` (default 0.8). Zhou & Farimani explicitly define *correlation time* as "timestep where Pearson correlation drops below a threshold (e.g. 0.8)", matching our implementation. |

**Removed metrics** (previously in our fork, now dropped to trim redundancy):

| Removed metric | Why |
|---|---|
| SobolevH1 | Mixed value+gradient units; not clearly interpretable for a stabilization thesis. NRMSE + spectral bins cover the same questions. |
| HighFreqEnergyRatio | Redundant with `binned_spectral_mse`. BSMSE measures actual per-band error; HFE ratio only tracks total energy in high-freq bands. |
| NRMSEAreaUnderCurve | Within a boundary-point rounding of `NRMSE_T=all`, which is already logged and more directly interpretable. |

**Upstream-Well metrics we do NOT use** (present in the clean clone but not re-exported in our fork's `__init__.py`): `HistogramW1` and `WindowedDTW` (trajectory-level metrics in `temporal.py`). These are from the Well paper but our fork trimmed them.

For the thesis:
- **CorrelationTime** is the cleanest citation — Zhou & Barati Farimani (2024) define it exactly as we implement it.
- **ValidRolloutLength** should be framed as the nRMSE-threshold analogue of the correlation-time / prediction-horizon concept from the same paper, e.g. *"We define valid rollout length as the time until per-step nRMSE exceeds a threshold, closely related to the correlation time of Zhou & Barati Farimani (2024)."*
- **ErrorGrowthRate** is our own formalisation. Cite Nayak & Goswami (2025) for the phenomenon of "rapid error accumulation" in autoregressive PDE surrogates, and state that λ is our quantification of it, e.g. *"We quantify rollout instability via an error growth rate λ, formalising the well-known phenomenon of autoregressive error accumulation (Nayak & Goswami, 2025)."*

## References

- **Trainer**: `the_well/benchmark/trainer/training.py`
- **Config**: `the_well/benchmark/configs/trainer/defaults.yaml`
- **W&B Analysis**: `docs/wandb_analysis.md`
- **Results**: `docs/wandb_analysis.md` → "Results Summary — Server Runs (2025-04)"
- **The Well paper**: Ohana et al., NeurIPS 2024, https://arxiv.org/abs/2412.00568 (Appendix E.3 "Metrics")
- **PDEBench paper**: Takamoto et al., NeurIPS 2022 D&B, https://arxiv.org/abs/2210.07182 (Appendix B "Detailed metrics description")
- **Correlation time definition** (and prediction-horizon framing): Zhou & Barati Farimani, "Predicting Change, Not States: An Alternate Framework for Neural PDE Surrogates", 2024 — source for `CorrelationTime` and conceptual anchor for `ValidRolloutLength`.
- **Error accumulation phenomenon**: Nayak & Goswami, "Data-Efficient Time-Dependent PDE Surrogates: Graph Neural Simulators vs Neural Operators", 2025 — source for the "rapid error accumulation" phenomenon that `ErrorGrowthRate` formalises.

# W&B Run Analysis — Workflow and Data Structure

## Project Details

- **Entity**: `simon-danninger-johannes-kepler-universit-t-linz`
- **Project**: `bsc_data_test`
- **API access**: credentials at `C:\Users\simon\_netrc`

## Fetching Runs via Python API

```python
import wandb
api = wandb.Api()
entity = "simon-danninger-johannes-kepler-universit-t-linz"

# List all runs in the project
runs = api.runs(f"{entity}/bsc_data_test")
for r in runs:
    print(r.id, r.name, r.state,
          r.config.get("temporal_bundle_size"),
          r.config.get("trainer", {}).get("noise_injection"))

# Fetch a specific run
r = api.run(f"{entity}/bsc_data_test/<run_id>")
```

> **Note**: The W&B entity name is NOT the GitHub username `annihilatorchess` — use the full JKU entity string above.

## Config Layout

Each run config mirrors the Hydra structure:

| Config path | Meaning |
|---|---|
| `r.config["seed"]` | Global RNG seed (added in recent commits) |
| `r.config["temporal_bundle_size"]` | Top-level bundle size (K); separate from trainer |
| `r.config["trainer"]["noise_injection"]` | Whether noise injection is active |
| `r.config["trainer"]["pushforward"]` | Whether pushforward training is active |
| `r.config["trainer"]["temporal_bundle_size"]` | Synced copy inside trainer config |

> Older runs (before seed support) show `seed=None`.

## Summary Metric Keys

Summary stores the **last logged value** per unique key. Keys follow the pattern:

```
{split}_{dataset_name}/{field}_{metric}_{time_range}
```

- `split`: `valid`, `rollout_valid`, `rollout_test`, `test`
- `field`: `density`, `velocity_x`, `velocity_y`, `pressure`, `full` (avg)
- `metric`: `NRMSE`, `VRMSE`, `MSE`, `LInfinity`, `HighFreqEnergyRatio`,
  `valid_rollout_length`, `valid_rollout_fraction`,
  `correlation_time`, `correlation_time_fraction`,
  `nrmse_auc`, `error_growth_rate`
- `time_range`: `T=all`, `T=0:19`, `T=19:38`, etc. (equal-width bins by default)

### Primary comparison metrics

```python
key_metrics = [
    "rollout_test_turbulent_radiative_layer_2D/full_VRMSE_T=all",
    "rollout_test_turbulent_radiative_layer_2D/density_NRMSE_T=all",
    "rollout_test_turbulent_radiative_layer_2D/velocity_x_NRMSE_T=all",
    "rollout_test_turbulent_radiative_layer_2D/velocity_y_NRMSE_T=all",
]
```

### Extended / summary metrics (per split)

These only appear in `rollout_valid` and `rollout_test` splits (rollout trajectories):

```python
extended_keys = [
    "rollout_test_{dataset}/{field}_valid_rollout_length",   # timesteps before nRMSE > 0.2
    "rollout_test_{dataset}/{field}_valid_rollout_fraction", # fraction of trajectory
    "rollout_test_{dataset}/{field}_correlation_time",       # timesteps before PearsonR < 0.8
    "rollout_test_{dataset}/{field}_nrmse_auc",              # trapz(nRMSE(t)) / T
    "rollout_test_{dataset}/{field}_error_growth_rate",      # λ from nRMSE ~ exp(λt)
]
```

> **Caveat**: The short `valid` split also logs these metrics (T=1 step), producing
> near-trivial values (nrmse_auc≈0, rollout_length≈1, correlation_time≈1).
> Always use `rollout_test_*` or `rollout_valid_*` for meaningful comparisons.

## History

For per-epoch curves (training loss, val loss over time):

```python
# Scan history without pandas
history = list(r.scan_history(keys=["train_loss", "valid", "_step"], page_size=1000))
for row in history:
    print(row.get("_step"), row.get("train_loss"))
```

## Run Frequency and What Gets Logged

| Event | Frequency | Metrics logged |
|---|---|---|
| Training step | Every epoch | `train_loss`, `epoch`, LR |
| Short validation | `val_frequency=1` (every epoch) | `valid_*` NRMSE/VRMSE/MSE per interval; extended metrics (but trivial, T=1) |
| Rollout validation | `rollout_val_frequency=2` | `rollout_valid_*` all metrics including extended |
| Test (final) | Once at epoch 100 | `rollout_test_*` all metrics |

> Prior to commit `e7fd9e8`, extended metrics (valid_rollout_length etc.) were gated
> by `full=epoch==max_epoch` in the rollout validation call, so they only appeared at
> epoch 100. This was fixed: rollout validation now always uses `full=True`.

## Results Summary & Aggregation

The extensive results previously documented here have been migrated and are continuously updated in the dedicated dataset-specific analysis files:
- `docs/performance_analysis/bsc_TRL_analysis.md` (The Well)
- `docs/performance_analysis/bsc_SWE_pdebench.md` (PDEBench)
- `docs/performance_analysis/BIG_SWE_input-4_analysis.md` (PDEBench)
- `docs/performance_analysis/BUR_DOWN_analysis.md` (PDEBench)

### Dataset Evaluation Conventions

To maintain strict comparability with the original source papers, we employ distinct evaluation conventions depending on the dataset family:

#### 1. The Well Datasets (e.g., `turbulent_radiative_layer_2D`, `active_matter`)
- **Primary Metric**: **VRMSE** (Variance Scaled Root Mean Squared Error). NRMSE is tracked but omitted from main summary tables.
- **Time-Averaged Intervals**: In addition to the full trajectory error (`T=all`), the time-averaged errors for intervals `T=6:12` and `T=13:30` are reported.
- **Weight Selection**: 
  - "Best Rollout Weights" strictly evaluates the checkpoint saved for lowest **rollout VRMSE**.
  - "Best One-Step Weights" strictly evaluates the checkpoint saved for lowest **one-step VRMSE**.

#### 2. PDEBench Datasets (e.g., `pdebench_swe`, `pdebench_1d_burgers_pdebench`)
- **Primary Metric**: **NRMSE** (Normalized Root Mean Squared Error). VRMSE is tracked but omitted from main summary tables to match the PDEBench paper's methodology.
- **Weight Selection**: 
  - "Best Rollout Weights" strictly evaluates the checkpoint saved for lowest **rollout NRMSE**.
  - "Best One-Step Weights" strictly evaluates the checkpoint saved for lowest **one-step NRMSE**.

#### Reporting Layout
All automated aggregator scripts generate tables that:
1. Group runs by ablation (Baseline, Pushforward, Temporal Bundling, BPTT, Noise Injection).
2. Segment performance by model architecture (`FNO`, `UNet`, `CNO`) and parameter size class (e.g., `SMALL`, `BIG`).
3. Include the **Average Selected Epoch** (`mean ± std`) directly as a column to track early-stopping behavior.
4. Prepend the official baseline metrics extracted from the source papers (The Well NeurIPS 2024, PDEBench NeurIPS 2022) for direct comparison.

## Common Pitfalls

- `valid_rollout_length` is a **float** (averaged over validation sequences), not an integer
- `correlation_time` unit is timesteps, NOT epochs
- Multiple duplicate-looking keys in summary come from different splits (`valid_*` vs `rollout_valid_*` vs `rollout_test_*`)
- `rollout_test_*` only exists for finished/complete runs; crashed runs only have `rollout_valid_*`
- Train MSE (normalized space) vs rollout MSE (physical space) are not comparable — density σ≈44 means physical MSE is ~1936× larger



## Advanced Metric Extraction (Best Checkpoints Strategy)

In later stages of the thesis, we transitioned to saving 4 distinct "best" checkpoints per run to better capture model performance:
- `best_one_step_nrmse`
- `best_one_step_vrmse`
- `best_rollout_nrmse`
- `best_rollout_vrmse`

### Weight Set vs. Evaluation Task

A common point of confusion is the difference between the **weights used** and the **evaluation task**. Each best weight set is evaluated on both a one-step task and a full rollout task.

The resulting summary keys in W&B follow this nested prefix structure:
`{task}_best_{weights}_{dataset}/full_{metric}_T=all`

| Key Component | Possible Values |
|---|---|
| `{task}` | `test` (One-Step), `rollout_test` (Full Rollout) |
| `{weights}` | `one_step_nrmse`, `one_step_vrmse`, `rollout_nrmse`, `rollout_vrmse` |
| `{metric}` | `NRMSE`, `VRMSE` |

**Example**: To find the rollout performance of the model that was best at one-step NRMSE:
`rollout_test_best_one_step_nrmse_turbulent_radiative_layer_2D/full_NRMSE_T=all`

### Lessons Learned for Data Fetching

1.  **Robust Numeric Parsing**: W&B summary values can occasionally be non-numeric (e.g., strings or NaNs) due to failed evaluations or logging errors. Always wrap metric extraction in a `float()` cast and handle `ValueError`/`TypeError`.
2.  **Dataset-Specific Keys**: The dataset name is hardcoded into the summary keys (e.g., `..._pdebench_swe/...`). Scripts must dynamically determine the dataset based on the W&B project or run configuration.
3.  **Group-Based Aggregation**: Rely on `run.group` for ablation grouping. It is much cleaner than parsing complex Hydra config dictionaries.
4.  **Standard Deviation (STD)**: When reporting results, always include the STD across seeds to distinguish between architectural improvements and noise.
5.  **Weight Sensitivity**: Architectures like **FNO** are extremely sensitive to the checkpoint selection. Their "best rollout" weights often perform worse on the test set than their "best one-step" weights, suggesting unstable rollout validation curves or overfitting to validation trajectories.
6.  **AMP and Runtime**: Hardware comparisons are only valid if **AMP** status is identical. AMP drastically reduces runtime and VRAM usage.

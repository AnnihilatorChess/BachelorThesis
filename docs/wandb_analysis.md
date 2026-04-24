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

## Results Summary — Server Runs (2025-04)

Dataset: `turbulent_radiative_layer_2D` | Architecture: `fno`

| Run ID | Config | State | VRMSE (test) | density NRMSE | vx NRMSE | vy NRMSE |
|---|---|---|---|---|---|---|
| `jrjj4433` | FNO baseline, seed=1 | finished | **3.94** | 0.395 | 0.630 | 2.025 |
| `zklzuanl` | FNO baseline, seed=2 | running | — | — | — | — |
| `vfy6bvy9` | bundle K=4, seed=1 | finished | **1.61** | 0.337 | 0.492 | 0.910 |
| `jguq751i` | bundle K=4, seed=2 | finished | **1.55** | 0.343 | 0.489 | 0.875 |
| `r5oe59zj` | noise σ=0.01, seed=1 | failed* | — | — | — | — |

*Noise injection run completed 100 training epochs but failed during final test evaluation (server-side GPU issue suspected).

### Extended metrics at test time

| Run | density corr_time | vx corr_time | density valid_rollout | vx valid_rollout | full nrmse_auc |
|---|---|---|---|---|---|
| FNO seed1 | 52.6/97 (54%) | 32.1/97 (33%) | 11.9/97 (12%) | 9.4/97 (10%) | 0.785 |
| bundle4 seed1 | 77.9/97 (80%) | 57.8/97 (60%) | 18.6/97 (19%) | 15.1/97 (16%) | 0.449 |
| bundle4 seed2 | 78.9/97 (81%) | 53.9/97 (56%) | 17.0/97 (18%) | 12.7/97 (13%) | 0.439 |

**Temporal bundling K=4** consistently improves all metrics:
- VRMSE reduced ~2.5× (3.94 → ~1.57 average)
- Correlation time ~+50% longer
- NRMSE AUC ~44% lower
- velocity_y improves most dramatically (2.03 → 0.89 NRMSE)

## Common Pitfalls

- `valid_rollout_length` is a **float** (averaged over validation sequences), not an integer
- `correlation_time` unit is timesteps, NOT epochs
- Multiple duplicate-looking keys in summary come from different splits (`valid_*` vs `rollout_valid_*` vs `rollout_test_*`)
- `rollout_test_*` only exists for finished/complete runs; crashed runs only have `rollout_valid_*`
- Train MSE (normalized space) vs rollout MSE (physical space) are not comparable — density σ≈44 means physical MSE is ~1936× larger

# PDEBench 2D Shallow Water Equations - Conversion to The Well Format

## Overview

This document describes the conversion of the PDEBench 2D Shallow Water Equations
(radial dam break) dataset into The Well's HDF5 format, enabling it to be used with
the existing training pipeline without modifications.

**Source**: PDEBench benchmark (Takamoto et al., NeurIPS 2022 Datasets & Benchmarks)
- Paper: https://arxiv.org/abs/2210.07182
- Data: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986

## Original PDEBench Format

### File
`datasets/PDEBench/2D_SWE/2D_rdb_NA_NA.h5` (6.2 GB)

### Physics
The 2D shallow water equations on a square domain with a radial dam break initial condition.
A circular bump of water (height=2.0) is placed at the centre; the surrounding water has height=1.0.
The dam radius varies per sample. Gravity is constant at 1.0. The solver is a Finite Volume Method (FVM)
using Clawpack's `pyclaw` with Roe solver and MC limiters. Boundary conditions are set via
`pyclaw.BC.wall` (homogeneous Neumann / reflecting) on all four domain edges.

### Structure
Each of the 1000 samples is stored as a separate HDF5 group (`0000/` through `0999/`):

```
0000/
  data:    shape=(101, 128, 128, 1)  float32   # (T, H, W, C=1)  water height
  grid/
    t:     shape=(101,)              float32   # [0.0, 0.01, ..., 1.0]
    x:     shape=(128,)              float32   # cell-centred in [-2.5, 2.5]
    y:     shape=(128,)              float32   # cell-centred in [-2.5, 2.5]
  attrs:
    config: "<YAML string>"   # contains sim params including dam_radius
```

The `config` attribute is a YAML string with simulation parameters nested under `sim`:
- `sim.dam_radius`: varies per sample (the conditioning parameter)
- `sim.gravity`: 1.0 (constant)
- `sim.inner_height`: 2.0 (constant)
- `sim.T_end`: 1.0
- `sim.n_time_steps`: 100 (+ initial condition = 101 snapshots)
- `sim.xdim`, `sim.ydim`: 128
- `sim.x_left`, `sim.x_right`: -2.5, 2.5
- `sim.y_bottom`, `sim.y_top`: -2.5, 2.5

### Key Properties
- All grids are identical across samples (verified)
- Only 1 channel: water height `h`
- The single varying parameter is `dam_radius`
- **Note on missing channels**: The SWE simulation internally computes all 5 state
  variables (h, u, v, hu, hv — height, velocities, and momenta). However, the
  `save_state_to_disk()` method in the PDEBench data generation code
  ([sim_radial_dam_break.py](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/src/sim_radial_dam_break.py))
  intentionally saves **only the height field `h`** to HDF5. The velocity and momentum
  arrays are discarded at write time. This is by design in the PDEBench codebase, not a
  download issue — there is no alternative file with all 3 channels.

## The Well Target Format

The Well expects a directory structure with rich HDF5 metadata:

```
pdebench_swe/
  stats.yaml                      # normalization statistics
  data/
    train/pdebench_swe.hdf5       # 800 trajectories
    valid/pdebench_swe.hdf5       # 100 trajectories
    test/pdebench_swe.hdf5        # 100 trajectories
```

Each HDF5 file has this internal structure:

```
/ (root)
  attrs: dataset_name, grid_type, n_spatial_dims, n_trajectories, simulation_parameters

  dimensions/
    attrs: spatial_dims = ["x", "y"]
    time:  (101,)   with attrs: sample_varying=False, time_varying=True
    x:     (128,)   with attrs: sample_varying=False, time_varying=False
    y:     (128,)   with attrs: sample_varying=False, time_varying=False

  t0_fields/
    attrs: field_names = ["h"]
    h:     (N, 101, 128, 128)  with attrs: sample_varying=True, time_varying=True, dim_varying=[True,True]

  t1_fields/   (empty)
  t2_fields/   (empty)

  scalars/
    attrs: field_names = ["dam_radius"]
    dam_radius: (N,)   with attrs: sample_varying=True, time_varying=False

  boundary_conditions/
    x_wall/  attrs: bc_type="WALL", associated_dims=["x"], ...
      mask: (128,) bool
    y_wall/  attrs: bc_type="WALL", associated_dims=["y"], ...
      mask: (128,) bool
```

## Conversion Steps

The conversion script (`scripts/convert_pdebench_swe.py`) performs these steps:

### 1. Split Assignment
The 1000 PDEBench samples are split into:
- **Train**: samples 0-799 (800 trajectories)
- **Valid**: samples 800-899 (100 trajectories)
- **Test**: samples 900-999 (100 trajectories)

### 2. Data Reshaping
PDEBench stores each sample separately as `(T, H, W, 1)`. The conversion:
- Squeezes the trailing channel dim: `(T, H, W, 1)` -> `(T, H, W)`
- Stacks all samples along a new leading dimension: `(N, T, H, W)`
- This matches The Well's convention where all trajectories share a single dataset

### 3. Grid Extraction
The time and spatial grids are extracted from the first sample (verified identical
across all samples) and stored once in `dimensions/`.

### 4. Parameter Extraction
`dam_radius` is parsed from each sample's YAML `config` attribute (under `sim.dam_radius`)
and stored as a sample-varying scalar in `scalars/dam_radius` with shape `(N,)`.

### 5. Boundary Conditions
The radial dam break uses **reflective (WALL)** boundary conditions on all domain edges.
These are stored as `x_wall` and `y_wall` boundary condition groups, each with a boolean
mask that is True at the first and last grid cells.

### 6. Metadata Attributes
All required Well attributes are synthesized:
- File-level: `dataset_name`, `grid_type`, `n_spatial_dims`, `n_trajectories`, `simulation_parameters`
- Per-field: `sample_varying`, `time_varying`, `dim_varying`
- Dimensions: `spatial_dims` group attribute

### 7. Normalization Statistics
`stats.yaml` is computed from the training split only:
- `mean`, `std`, `rms` of the field `h` across all samples/timesteps/spatial points
- `mean_delta`, `std_delta`, `rms_delta` of temporal differences `h(t+1) - h(t)`

Computed values from the current training split:
```yaml
mean:  {h: 1.0327}
std:   {h: 0.1141}
rms:   {h: 1.0388}
mean_delta:  {h: -2.2266e-12}   # ≈ 0 — no drift bias
std_delta:   {h: 7.8778e-03}
rms_delta:   {h: 7.8778e-03}
```
The near-zero `mean_delta` confirms the field is approximately conservative over time. The small `std_delta` (~0.8% of the field std) indicates slow temporal dynamics.

## Additional Files Created

### `the_well/data/utils.py`
Added `"pdebench_swe"` to `WELL_DATASETS` list so that `WellDataset` recognises it.

### `the_well/benchmark/configs/data/pdebench_swe.yaml`
Hydra data config for training:
```yaml
_target_: the_well.data.WellDataModule
batch_size: 32
well_base_path: /mnt/gpuxl/polymathic/the_well/datasets/
well_dataset_name: pdebench_swe
use_normalization: True
min_dt_stride: 1
max_dt_stride: 1
```
Note: `well_base_path` is currently hardcoded for the gpuxl server. Override via `data.well_base_path=...` or the `server` config when running locally.

## Verification

Loading the converted dataset with WellDataset succeeds:

```python
from the_well.data.datasets import WellDataset

ds = WellDataset(
    well_base_path="datasets",
    well_dataset_name="pdebench_swe",
    well_split_name="train",
)
# Length: 80000 (800 trajectories x 100 windows)
# Sample: input_fields (1,128,128,1), output_fields (1,128,128,1),
#         constant_scalars (1,), boundary_conditions (2,2),
#         space_grid (128,128,2), time grids (1,)
```

The `constant_scalars` tensor contains `dam_radius`, making it available for FiLM conditioning.

## Notes for Training

To train on this dataset:
```bash
python the_well/benchmark/train.py experiment=unet_classic server=local data=pdebench_swe
```

Ensure `server.well_base_path` points to the directory containing `pdebench_swe/`.

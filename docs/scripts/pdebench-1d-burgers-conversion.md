# PDEBench 1D Burgers - Conversion to The Well Format

## Overview

This document describes the conversion of the PDEBench 1D Burgers dataset into The Well's HDF5 format.

**Source**: PDEBench benchmark (Takamoto et al., NeurIPS 2022)
- Data: `1D_Burgers_Sols_Nu0.001.hdf5`

> **Note:** The conversion script currently uses **only** the $\nu=0.001$ file. The $\nu=0.01$ file was dropped; mixing regimes degraded training. To re-enable it, add back to `SOURCE_FILES` in `scripts/convert_pdebench_1d_burgers.py`.

## Original PDEBench Format

The $\nu=0.001$ parameter is distributed as a single 7.7 GB HDF5 file containing 10,000 samples.

### Structure
Inside the file:
- `tensor`: `(10000, 201, 1024)` - the velocity field $u(t,x)$
- `t-coordinate`: `(202,)` - time steps from 0.0 to 2.01
- `x-coordinate`: `(1024,)` - spatial grid from 0.0 to ~1.0

*Note:* `t-coordinate` has 202 elements, but the data `tensor` only has 201 steps. The conversion script extracts the first 201 coordinates (0.0 to 2.0) to match the data shape.

## The Well Target Format

The conversion script (`scripts/convert_pdebench_1d_burgers.py`) writes a single-parameter dataset.

### 1. Viscosity Scalar
Even with only one viscosity value, the script stores $\nu$ as a `sample_varying` `scalar` named `nu` for compatibility with The Well format. This leaves the door open to re-add multiple viscosities later without a schema change.

### 2. Splitting Strategy
The dataset consists of 10,000 trajectories, split sequentially:
- **Train**: 8,000
- **Valid**: 1,000
- **Test**: 1,000

### 3. File Chunking
For I/O performance (specifically, to avoid HDF5 global lock contention with parallel DataLoader workers), the final split files are chunked into smaller files. The script defaults to creating **16 splits** per dataset (train/valid/test) to maximize I/O parallelism with a high number of data workers.
e.g., `pdebench_1d_burgers_00.hdf5` to `pdebench_1d_burgers_15.hdf5`.

### 4. Boundary Conditions
The 1D Burgers equation from PDEBench uses periodic boundary conditions. This is reflected in the metadata as `PERIODIC` on the spatial dimension `x` via `x_periodic` mask.

## Execution

To run the conversion:
```bash
python scripts/convert_pdebench_1d_burgers.py
```
This requires `1D_Burgers_Sols_Nu0.001.hdf5` to be present in `datasets/PDEBench/1D_Burgers/`.

## Normalization Statistics
The script automatically generates `stats.yaml` from the `train` split, containing `mean`, `std`, and `rms` for the velocity field `u` and its temporal derivatives.

## Training Configuration
The data configuration already exists at `the_well/benchmark/configs/data/pdebench_1d_burgers.yaml`:
```yaml
_target_: the_well.data.WellDataModule
batch_size: 1024  # 1D models can support large batch sizes
well_base_path: /mnt/gpuxl/polymathic/the_well/datasets/
well_dataset_name: pdebench_1d_burgers
use_normalization: True
min_dt_stride: 1
max_dt_stride: 1
```
Override `data.well_base_path=...` (or use a `server=...` config) when training locally.

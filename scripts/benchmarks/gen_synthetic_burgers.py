"""Generate a synthetic 1D Burgers HDF5 file matching the Well format.

Layout mirrors scripts/convert_pdebench_1d_burgers.py exactly — same groups,
same `t0_fields/u` layout (contiguous, no chunking), same `scalars/nu`, same
boundary conditions. Only difference: data is random noise, and we default
to fewer trajectories so it fits comfortably on a laptop.

Usage:
    python scripts/benchmarks/gen_synthetic_burgers.py --n_traj 2000
"""

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

# Use the real dataset name so it passes WellDataset's allowlist.
OUT_ROOT = Path("datasets/pdebench_1d_burgers")
DATASET_NAME = "pdebench_1d_burgers"
N_STEPS = 201
N_X = 1024
DT = 0.01
X_MAX = 1.0
NU = 0.001


def write_split(split_dir: Path, n_traj: int, seed: int):
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{DATASET_NAME}.hdf5"
    if out_path.exists():
        out_path.unlink()

    rng = np.random.default_rng(seed)
    t_grid = np.arange(N_STEPS, dtype=np.float32) * DT
    x_grid = np.linspace(0, X_MAX, N_X, endpoint=False, dtype=np.float32)

    with h5py.File(out_path, "w", libver="latest") as f:
        f.attrs["dataset_name"] = DATASET_NAME
        f.attrs["n_spatial_dims"] = 1
        f.attrs["n_trajectories"] = n_traj
        f.attrs["grid_type"] = "cartesian"
        f.attrs["simulation_parameters"] = []

        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x"]
        ds_t = dims.create_dataset("time", data=t_grid)
        ds_t.attrs["sample_varying"] = False
        ds_t.attrs["time_varying"] = True
        ds_x = dims.create_dataset("x", data=x_grid)
        ds_x.attrs["sample_varying"] = False
        ds_x.attrs["time_varying"] = False

        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = ["u"]
        # Same as conversion script: contiguous, no chunking
        ds_u = t0.create_dataset(
            "u",
            shape=(n_traj, N_STEPS, N_X),
            dtype=np.float32,
        )
        ds_u.attrs["sample_varying"] = True
        ds_u.attrs["time_varying"] = True
        ds_u.attrs["dim_varying"] = [True]

        # Fill with random data in reasonable chunks
        BLOCK = 200
        for start in range(0, n_traj, BLOCK):
            end = min(start + BLOCK, n_traj)
            ds_u[start:end] = rng.standard_normal(
                (end - start, N_STEPS, N_X), dtype=np.float32
            )

        scalars = f.create_group("scalars")
        scalars.attrs["field_names"] = ["nu"]
        ds_nu = scalars.create_dataset(
            "nu", shape=(n_traj,), dtype=np.float32
        )
        ds_nu[:] = NU
        ds_nu.attrs["sample_varying"] = True
        ds_nu.attrs["time_varying"] = False

        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = []
        t2 = f.create_group("t2_fields")
        t2.attrs["field_names"] = []

        bc = f.create_group("boundary_conditions")
        bc_grp = bc.create_group("x_periodic")
        bc_grp.attrs["bc_type"] = "PERIODIC"
        bc_grp.attrs["associated_dims"] = ["x"]
        bc_grp.attrs["associated_fields"] = []
        bc_grp.attrs["sample_varying"] = False
        bc_grp.attrs["time_varying"] = False
        mask = np.zeros(N_X, dtype=bool)
        mask[0] = True
        mask[-1] = True
        bc_grp.create_dataset("mask", data=mask)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {out_path} ({size_mb:.1f} MB, {n_traj} traj)")


def write_stats(out_root: Path):
    stats = {
        "mean": {"u": 0.0},
        "std": {"u": 1.0},
        "rms": {"u": 1.0},
        "mean_delta": {"u": 0.0},
        "std_delta": {"u": 1.4142},
        "rms_delta": {"u": 1.4142},
    }
    (out_root / "stats.yaml").write_text(
        yaml.dump(stats, default_flow_style=False, sort_keys=False)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_traj", type=int, default=2000,
                        help="number of training trajectories (default 2000 ≈ 1.6 GB)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    if args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True)

    write_split(args.out_root / "data" / "train", args.n_traj, args.seed)
    write_split(args.out_root / "data" / "valid", max(args.n_traj // 8, 100), args.seed + 1)
    write_split(args.out_root / "data" / "test", max(args.n_traj // 8, 100), args.seed + 2)
    write_stats(args.out_root)
    print(f"done: {args.out_root}")


if __name__ == "__main__":
    main()

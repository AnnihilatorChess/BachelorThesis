"""
Convert PDEBench 1D Burgers dataset to The Well HDF5 format,
and split each split into multiple files for parallel DataLoader I/O.

PDEBench source: https://arxiv.org/abs/2210.07182
  - Files: 1D_Burgers_Sols_Nu0.01.hdf5, 1D_Burgers_Sols_Nu0.001.hdf5
  - 10000 samples per file
  - 'tensor' shape: (10000, 201, 1024) = (N, T, X)
  - Grid: t in [0, 2.0] (dt=0.01, 201 steps), x in [0, 1) (1024 points)
  - Varying parameter: nu (viscosity), extracted from filename
  - Boundary conditions: PERIODIC

The Well target format:
  - Groups: dimensions/, t0_fields/, t1_fields/, t2_fields/, scalars/, boundary_conditions/
  - Directory structure: <dataset>/data/{train,valid,test}/<file>.hdf5

PDEBench-protocol downsampling:
  PDEBench's ``config_Bgs.yaml`` applies ``reduced_resolution=4`` (1024 → 256
  spatial) and ``reduced_resolution_t=5`` (201 → 41 temporal) before training,
  via plain ``data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]``
  slicing in their dataloader (``pdebench/models/fno/utils.py``). Pass these
  flags below to bake the same naive subsampling into the converted HDF5
  files, producing a separate output directory ``pdebench_1d_burgers_pdebench``.
  The (untouched) full-resolution variant is the one the rest of the thesis
  uses; the downsampled variant exists solely for PDEBench-protocol parity
  experiments (``experiment/{fno,unet}_pdebench_burgers.yaml``).

Usage:
    # full resolution (default; writes datasets/pdebench_1d_burgers/)
    python scripts/convert_pdebench_1d_burgers.py

    # PDEBench-protocol parity (writes datasets/pdebench_1d_burgers_pdebench/)
    python scripts/convert_pdebench_1d_burgers.py --variant pdebench

    # explicit subsampling factors
    python scripts/convert_pdebench_1d_burgers.py \
        --reduced_resolution 4 --reduced_resolution_t 5
"""

import argparse
import os
import sys
import yaml
import h5py
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
SOURCE_FILES = [
    (Path("datasets/PDEBench/1D_Burgers/1D_Burgers_Sols_Nu0.001.hdf5"), 0.001),
]
# Default (full-resolution) output. The downsampled variant lives at
# ``datasets/pdebench_1d_burgers_pdebench/`` (see ``resolve_output_layout``).
OUTPUT_BASE = Path("datasets/pdebench_1d_burgers")
DATASET_NAME = "pdebench_1d_burgers"

# Train / valid / test split per file (10000 total samples per file)
SPLIT_RANGES = {
    "train": (0, 8000),      # 8000 samples total
    "valid": (8000, 9000),   # 1000 samples total
    "test":  (9000, 10000),  # 1000 samples total
}


# ──────────────────────────────────────────────────────────────────────
# Conversion helpers
# ──────────────────────────────────────────────────────────────────────

def convert_split(
    split: str,
    start: int,
    end: int,
    output_base: Path,
    dataset_name: str,
    repo_root: Path,
    reduced_resolution: int = 1,
    reduced_resolution_t: int = 1,
):
    """Convert a range of PDEBench samples from multiple files into a single Well-format HDF5 file.

    ``reduced_resolution`` and ``reduced_resolution_t`` apply naive ``[::n]``
    subsampling to the spatial and temporal axes respectively, mirroring
    PDEBench's dataloader-time downsampling.
    """
    n_traj_per_file = end - start
    n_files = len(SOURCE_FILES)
    n_traj = n_traj_per_file * n_files

    out_dir = output_base / "data" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}.hdf5"

    print(f"  Converting {split}: {n_traj} total trajectories -> {out_path}")

    # Read grids from the first file
    first_file_path, _ = SOURCE_FILES[0]
    with h5py.File(repo_root / first_file_path, "r") as src:
        # Take only first 201 to match tensor shape, then apply temporal stride.
        t = src["t-coordinate"][:201:reduced_resolution_t].astype(np.float32)
        x = src["x-coordinate"][::reduced_resolution].astype(np.float32)

    n_steps = len(t)   # 201 // reduced_resolution_t (rounded up)
    nx = len(x)        # 1024 // reduced_resolution (rounded up)

    if reduced_resolution > 1 or reduced_resolution_t > 1:
        print(
            f"    PDEBench-protocol downsampling: "
            f"reduced_resolution={reduced_resolution}, "
            f"reduced_resolution_t={reduced_resolution_t} "
            f"(shape per traj: ({n_steps}, {nx}))"
        )

    with h5py.File(out_path, "w") as dst:
        # ── File-level attributes ──────────────────────────────────
        dst.attrs["dataset_name"] = dataset_name
        dst.attrs["grid_type"] = "cartesian"
        dst.attrs["n_spatial_dims"] = 1
        dst.attrs["n_trajectories"] = n_traj
        dst.attrs["simulation_parameters"] = ["nu"]
        # Record provenance of the subsampling so we can tell variants apart.
        dst.attrs["reduced_resolution"] = reduced_resolution
        dst.attrs["reduced_resolution_t"] = reduced_resolution_t

        # ── Dimensions ─────────────────────────────────────────────
        dims = dst.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x"]

        ds_time = dims.create_dataset("time", data=t)
        ds_time.attrs["sample_varying"] = False
        ds_time.attrs["time_varying"] = True

        ds_x = dims.create_dataset("x", data=x)
        ds_x.attrs["sample_varying"] = False
        ds_x.attrs["time_varying"] = False

        # ── t0_fields (scalar fields — velocity u) ──────────────
        t0 = dst.create_group("t0_fields")
        t0.attrs["field_names"] = ["u"]

        ds_u = t0.create_dataset(
            "u",
            shape=(n_traj, n_steps, nx),
            dtype=np.float32,
            # No chunking — contiguous layout matches The Well's own datasets.
            # HDF5 seeks to the exact byte offset for any timestep window, so DataLoader
            # workers read only what they actually need rather than the full trajectory.
        )
        ds_u.attrs["sample_varying"] = True
        ds_u.attrs["time_varying"] = True
        ds_u.attrs["dim_varying"] = [True]

        # ── Scalars ────────────────────────────────────────────────
        scalars = dst.create_group("scalars")
        scalars.attrs["field_names"] = ["nu"]
        ds_nu = scalars.create_dataset("nu", shape=(n_traj,), dtype=np.float32)
        ds_nu.attrs["sample_varying"] = True
        ds_nu.attrs["time_varying"] = False

        # ── Boundary conditions ────────────────────────────────────
        bc = dst.create_group("boundary_conditions")
        bc_grp = bc.create_group("x_periodic")
        bc_grp.attrs["bc_type"] = "PERIODIC"
        bc_grp.attrs["associated_dims"] = ["x"]
        bc_grp.attrs["associated_fields"] = []
        bc_grp.attrs["sample_varying"] = False
        bc_grp.attrs["time_varying"] = False
        mask = np.zeros(nx, dtype=bool)
        mask[0] = True
        mask[-1] = True
        bc_grp.create_dataset("mask", data=mask)

        # ── Copy Data ──────────────────────────────────────────────
        offset = 0
        for src_rel_path, nu_val in SOURCE_FILES:
            src_path = repo_root / src_rel_path
            print(f"    Reading {src_path.name} (nu={nu_val})")
            with h5py.File(src_path, "r") as src:
                # Apply the same naive ``[::n]`` subsampling PDEBench uses in
                # ``pdebench/models/fno/utils.py`` (see Burgers/Adv configs).
                data = src["tensor"][
                    start:end, ::reduced_resolution_t, ::reduced_resolution
                ]
                ds_u[offset : offset + n_traj_per_file] = data
                ds_nu[offset : offset + n_traj_per_file] = nu_val
                offset += n_traj_per_file

        # ── t1_fields (empty) ──────────────────────────────────────
        t1 = dst.create_group("t1_fields")
        t1.attrs["field_names"] = []

        # ── t2_fields (empty) ──────────────────────────────────────
        t2 = dst.create_group("t2_fields")
        t2.attrs["field_names"] = []

    print(f"  OK: {split} done: {n_traj} trajectories, shape ({n_traj}, {n_steps}, {nx})")
    return out_path


# ──────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────

def compute_stats(output_base: Path, dataset_name: str = DATASET_NAME):
    """Compute normalization statistics from the training split and write stats.yaml."""
    train_path = output_base / "data" / "train" / f"{dataset_name}.hdf5"

    print("\nComputing normalization statistics from training data...")
    with h5py.File(train_path, "r") as f:
        u_data = f["t0_fields"]["u"][:]  # (N, T, X)

    mean_u = float(np.mean(u_data))
    std_u = float(np.std(u_data))
    rms_u = float(np.sqrt(np.mean(u_data ** 2)))

    deltas = u_data[:, 1:, :] - u_data[:, :-1, :]
    mean_delta_u = float(np.mean(deltas))
    std_delta_u = float(np.std(deltas))
    rms_delta_u = float(np.sqrt(np.mean(deltas ** 2)))

    stats = {
        "mean":       {"u": mean_u},
        "std":        {"u": std_u},
        "rms":        {"u": rms_u},
        "mean_delta": {"u": mean_delta_u},
        "std_delta":  {"u": std_delta_u},
        "rms_delta":  {"u": rms_delta_u},
    }

    stats_path = output_base / "stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

    print(f"  OK: Stats written to {stats_path}")
    print(f"    u: mean={mean_u:.4f}, std={std_u:.4f}, rms={rms_u:.4f}")
    print(f"    du: mean={mean_delta_u:.6f}, std={std_delta_u:.6f}, rms={rms_delta_u:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────────────────

def split_file(src_path: Path, n_splits: int):
    """Split a single Well-format HDF5 file into n_splits smaller files."""
    src_path = Path(src_path)
    out_dir = src_path.parent

    with h5py.File(src_path, "r") as src:
        n_traj = int(src.attrs["n_trajectories"])
        chunk_size = int(np.ceil(n_traj / n_splits))

        traj_datasets = {}
        shared_datasets = {}

        def collect(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.shape and obj.shape[0] == n_traj:
                traj_datasets[name] = obj
            else:
                shared_datasets[name] = obj

        src.visititems(collect)

        created = []
        for i in range(n_splits):
            start = i * chunk_size
            end = min(start + chunk_size, n_traj)
            if start >= n_traj:
                break

            out_name = f"{src_path.stem}_{i:02d}.hdf5"
            out_path = out_dir / out_name

            with h5py.File(out_path, "w") as dst:
                for k, v in src.attrs.items():
                    dst.attrs[k] = v
                dst.attrs["n_trajectories"] = end - start

                def copy_group_attrs(name, obj):
                    if isinstance(obj, h5py.Group):
                        grp = dst.require_group(name)
                        for k, v in obj.attrs.items():
                            grp.attrs[k] = v
                src.visititems(copy_group_attrs)

                for name, ds in shared_datasets.items():
                    grp_path = "/".join(name.split("/")[:-1])
                    if grp_path and grp_path not in dst:
                        dst.require_group(grp_path)
                    src.copy(ds, dst, name=name)

                for name, ds in traj_datasets.items():
                    data = ds[start:end]
                    grp_path = "/".join(name.split("/")[:-1])
                    if grp_path and grp_path not in dst:
                        dst.require_group(grp_path)
                    src_chunks = ds.chunks
                    chunks = (min(1, data.shape[0]),) + src_chunks[1:] if src_chunks else None
                    new_ds = dst.create_dataset(name, data=data, chunks=chunks)
                    for k, v in ds.attrs.items():
                        new_ds.attrs[k] = v

            size_mb = out_path.stat().st_size / 1e6
            print(f"    {out_name}: {end - start} traj, {size_mb:.0f} MB")
            created.append(out_path)

    src_path.unlink()
    print(f"    Removed original: {src_path.name}")
    return created


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def resolve_output_layout(reduced_resolution: int, reduced_resolution_t: int):
    """Pick the output directory and dataset name based on the subsampling factors.

    Full resolution (1, 1) → ``datasets/pdebench_1d_burgers``  (existing data,
    used by the rest of the thesis).
    Anything else → ``datasets/pdebench_1d_burgers_pdebench`` (PDEBench-protocol
    parity dataset). Storing both side-by-side lets the thesis-wide
    "no downsampling" rule and the parity experiments coexist without
    overwriting each other.
    """
    if reduced_resolution == 1 and reduced_resolution_t == 1:
        return Path("datasets/pdebench_1d_burgers"), "pdebench_1d_burgers"
    return (
        Path("datasets/pdebench_1d_burgers_pdebench"),
        "pdebench_1d_burgers_pdebench",
    )


def main():
    parser = argparse.ArgumentParser(description="Convert PDEBench 1D Burgers to The Well format")
    parser.add_argument("--n_splits", type=int, default=16,
                        help="Number of files to split each split into (default: 16).")
    parser.add_argument("--no_split", action="store_true",
                        help="Skip the splitting step entirely.")
    parser.add_argument("--variant", choices=("full", "pdebench"), default="full",
                        help=("'full' keeps the original 201 x 1024 resolution. "
                              "'pdebench' applies PDEBench's reduced_resolution=4 "
                              "+ reduced_resolution_t=5 -> 41 x 256 (writes to a "
                              "separate output directory, default 'full' is safe "
                              "to re-run without disturbing the parity dataset)."))
    parser.add_argument("--reduced_resolution", type=int, default=None,
                        help="Naive [::n] spatial subsampling. Overrides --variant.")
    parser.add_argument("--reduced_resolution_t", type=int, default=None,
                        help="Naive [::n] temporal subsampling. Overrides --variant.")
    args = parser.parse_args()

    if args.no_split:
        args.n_splits = 1

    # Resolve subsampling factors. Explicit flags trump --variant.
    if args.reduced_resolution is not None or args.reduced_resolution_t is not None:
        reduced_r = args.reduced_resolution or 1
        reduced_t = args.reduced_resolution_t or 1
    elif args.variant == "pdebench":
        reduced_r, reduced_t = 4, 5
    else:
        reduced_r, reduced_t = 1, 1

    # Find repo_root dynamically
    if Path("datasets/PDEBench").exists():
        repo_root = Path.cwd()
    elif Path("../../datasets/PDEBench").exists():
        repo_root = Path.cwd() / "../.."
    else:
        # Fallback to local script location
        repo_root = Path(__file__).resolve().parent.parent

    output_rel, dataset_name = resolve_output_layout(reduced_r, reduced_t)
    output_base = repo_root / output_rel

    print(f"Variant: reduced_resolution={reduced_r}, reduced_resolution_t={reduced_t}")
    print(f"Output:  {output_base}  (dataset_name='{dataset_name}')")
    print()

    # Verify source files
    for src_rel_path, nu_val in SOURCE_FILES:
        src_path = repo_root / src_rel_path
        if not src_path.exists():
            print(f"ERROR: Source file not found: {src_path}")
            sys.exit(1)

    # ── Step 1: Convert ───────────────────────────────────────────
    for split, (start, end) in SPLIT_RANGES.items():
        convert_split(
            split, start, end, output_base, dataset_name, repo_root,
            reduced_resolution=reduced_r,
            reduced_resolution_t=reduced_t,
        )

    # ── Step 2: Normalization stats ───────────────────────────────
    compute_stats(output_base, dataset_name)

    # ── Step 3: Split into multiple files ─────────────────────────
    if args.n_splits > 1:
        print(f"\nSplitting each file into {args.n_splits} parts...")
        for split in SPLIT_RANGES:
            single_file = output_base / "data" / split / f"{dataset_name}.hdf5"
            if single_file.exists():
                print(f"  {split}/")
                split_file(single_file, args.n_splits)

    print("\nOK: Conversion complete!")

if __name__ == "__main__":
    main()

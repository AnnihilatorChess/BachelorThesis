"""
Convert PDEBench 2D Shallow Water Equations (radial dam break) dataset
to The Well HDF5 format, then split each split into multiple files for
parallel DataLoader I/O (avoids HDF5 lock contention with many workers).

PDEBench source: https://arxiv.org/abs/2210.07182
  - File: 2D_rdb_NA_NA.h5
  - 1000 samples, each (101, 128, 128, 1) = (T, H, W, C=water_height)
  - Grid: t in [0,1] (dt=0.01), x/y in [-2.5, 2.5] (128 cell-centred)
  - Per-sample varying parameter: dam_radius
  - Constants: gravity=1.0, inner_height=2.0
  - Boundary conditions: reflective (WALL) — radial dam break on square domain

The Well target format:
  - All trajectories stacked along a leading sample dimension
  - Groups: dimensions/, t0_fields/, t1_fields/, t2_fields/, scalars/, boundary_conditions/
  - Rich per-dataset and per-field attributes
  - Directory structure: <dataset>/data/{train,valid,test}/<file>.hdf5

Usage:
    python scripts/convert_pdebench_swe.py                 # convert + split into 8 files
    python scripts/convert_pdebench_swe.py --n_splits 4    # split into 4 files instead
    python scripts/convert_pdebench_swe.py --no_split      # convert only, single file per split
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
PDEBENCH_FILE = Path(r"datasets/PDEBench/2D_SWE/2D_rdb_NA_NA.h5")
OUTPUT_BASE = Path(r"datasets/pdebench_swe")
DATASET_NAME = "pdebench_swe"

# Train / valid / test split (1000 total samples, indices 0-based)
# PDEBench samples are randomly ordered w.r.t. dam_radius, so a sequential
# split gives approximately equal parameter distributions across all three splits.
SPLIT_RANGES = {
    "train": (0, 800),     # 800 samples
    "valid": (800, 900),   # 100 samples
    "test":  (900, 1000),  # 100 samples
}


# ──────────────────────────────────────────────────────────────────────
# Conversion helpers
# ──────────────────────────────────────────────────────────────────────

def read_pdebench_metadata(src: h5py.File):
    """Read grid and parameter info from the first sample."""
    sample = src["0000"]
    t = sample["grid"]["t"][:]
    x = sample["grid"]["x"][:]
    y = sample["grid"]["y"][:]

    config_str = sample.attrs["config"]
    if isinstance(config_str, bytes):
        config_str = config_str.decode("utf-8")
    config = yaml.safe_load(config_str)
    sim = config.get("sim", config)

    return {
        "t": t.astype(np.float32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "gravity": float(sim.get("gravity", 1.0)),
        "inner_height": float(sim.get("inner_height", 2.0)),
    }


def read_dam_radius(src: h5py.File, idx: int) -> float:
    """Extract dam_radius from a sample's config attribute."""
    key = f"{idx:04d}"
    config_str = src[key].attrs["config"]
    if isinstance(config_str, bytes):
        config_str = config_str.decode("utf-8")
    config = yaml.safe_load(config_str)
    sim = config.get("sim", config)
    return float(sim["dam_radius"])


def convert_split(src: h5py.File, meta: dict, split: str, start: int, end: int,
                  output_base: Path):
    """Convert a range of PDEBench samples into a single Well-format HDF5 file."""
    n_traj = end - start
    n_steps = len(meta["t"])   # 101
    nx = len(meta["x"])        # 128
    ny = len(meta["y"])        # 128

    out_dir = output_base / "data" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{DATASET_NAME}.hdf5"

    print(f"  Converting {split}: samples {start}-{end-1} -> {out_path}")

    with h5py.File(out_path, "w") as dst:
        # ── File-level attributes ──────────────────────────────────
        dst.attrs["dataset_name"] = DATASET_NAME
        dst.attrs["grid_type"] = "cartesian"
        dst.attrs["n_spatial_dims"] = 2
        dst.attrs["n_trajectories"] = n_traj
        dst.attrs["simulation_parameters"] = ["dam_radius"]

        # ── Dimensions ─────────────────────────────────────────────
        dims = dst.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]

        ds_time = dims.create_dataset("time", data=meta["t"])
        ds_time.attrs["sample_varying"] = False
        ds_time.attrs["time_varying"] = True

        ds_x = dims.create_dataset("x", data=meta["x"])
        ds_x.attrs["sample_varying"] = False
        ds_x.attrs["time_varying"] = False

        ds_y = dims.create_dataset("y", data=meta["y"])
        ds_y.attrs["sample_varying"] = False
        ds_y.attrs["time_varying"] = False

        # ── t0_fields (scalar fields — water height) ──────────────
        t0 = dst.create_group("t0_fields")
        t0.attrs["field_names"] = ["h"]

        ds_h = t0.create_dataset(
            "h",
            shape=(n_traj, n_steps, nx, ny),
            dtype=np.float32,
            chunks=(1, n_steps, nx, ny),   # one chunk per trajectory for fast access
        )
        ds_h.attrs["sample_varying"] = True
        ds_h.attrs["time_varying"] = True
        ds_h.attrs["dim_varying"] = [True, True]

        for i, src_idx in enumerate(range(start, end)):
            key = f"{src_idx:04d}"
            data = src[key]["data"][:, :, :, 0]  # (101, 128, 128, 1) -> (101, 128, 128)
            ds_h[i] = data
            if (i + 1) % 100 == 0:
                print(f"    {split}: {i+1}/{n_traj} trajectories copied")

        # ── t1_fields (empty — no vector fields) ──────────────────
        t1 = dst.create_group("t1_fields")
        t1.attrs["field_names"] = []

        # ── t2_fields (empty — no rank-2 tensor fields) ───────────
        t2 = dst.create_group("t2_fields")
        t2.attrs["field_names"] = []

        # ── Scalars ────────────────────────────────────────────────
        scalars = dst.create_group("scalars")
        scalars.attrs["field_names"] = ["dam_radius"]

        dam_radii = np.array(
            [read_dam_radius(src, idx) for idx in range(start, end)],
            dtype=np.float32,
        )
        ds_dr = scalars.create_dataset("dam_radius", data=dam_radii)
        ds_dr.attrs["sample_varying"] = True
        ds_dr.attrs["time_varying"] = False

        # ── Boundary conditions ────────────────────────────────────
        bc = dst.create_group("boundary_conditions")

        for dim_name in ["x", "y"]:
            bc_grp = bc.create_group(f"{dim_name}_wall")
            bc_grp.attrs["bc_type"] = "WALL"
            bc_grp.attrs["associated_dims"] = [dim_name]
            bc_grp.attrs["associated_fields"] = []
            bc_grp.attrs["sample_varying"] = False
            bc_grp.attrs["time_varying"] = False
            mask = np.zeros(nx if dim_name == "x" else ny, dtype=bool)
            mask[0] = True
            mask[-1] = True
            bc_grp.create_dataset("mask", data=mask)

    print(f"  OK: {split} done: {n_traj} trajectories, shape ({n_traj}, {n_steps}, {nx}, {ny})")
    return out_path


# ──────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────

def compute_stats(output_base: Path):
    """Compute normalization statistics from the training split and write stats.yaml.

    Always reads from the single consolidated train file (before or after splitting).
    Stats are computed from training data only to avoid test/val contamination.
    """
    train_path = output_base / "data" / "train" / f"{DATASET_NAME}.hdf5"

    print("\nComputing normalization statistics from training data...")
    with h5py.File(train_path, "r") as f:
        h_data = f["t0_fields"]["h"][:]  # (N, T, H, W)

    mean_h = float(np.mean(h_data))
    std_h = float(np.std(h_data))
    rms_h = float(np.sqrt(np.mean(h_data ** 2)))

    deltas = h_data[:, 1:, :, :] - h_data[:, :-1, :, :]
    mean_delta_h = float(np.mean(deltas))
    std_delta_h = float(np.std(deltas))
    rms_delta_h = float(np.sqrt(np.mean(deltas ** 2)))

    stats = {
        "mean":       {"h": mean_h},
        "std":        {"h": std_h},
        "rms":        {"h": rms_h},
        "mean_delta": {"h": mean_delta_h},
        "std_delta":  {"h": std_delta_h},
        "rms_delta":  {"h": rms_delta_h},
    }

    stats_path = output_base / "stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

    print(f"  OK: Stats written to {stats_path}")
    print(f"    h: mean={mean_h:.4f}, std={std_h:.4f}, rms={rms_h:.4f}")
    print(f"    dh: mean={mean_delta_h:.6f}, std={std_delta_h:.6f}, rms={rms_delta_h:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────────────────

def split_file(src_path: Path, n_splits: int):
    """Split a single Well-format HDF5 file into n_splits smaller files.

    Each output file is an independently valid Well HDF5 file with correct
    root-level and group-level attributes (including n_trajectories reflecting
    the actual count in that file). Chunk layout is preserved from the source
    so per-sample read cost is identical. No compression is applied.

    The original file is deleted after successful splitting.
    """
    src_path = Path(src_path)
    out_dir = src_path.parent

    with h5py.File(src_path, "r") as src:
        n_traj = int(src.attrs["n_trajectories"])
        chunk_size = int(np.ceil(n_traj / n_splits))

        # Classify datasets: trajectory-indexed vs shared
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
                # Root-level attributes — n_trajectories reflects this file's count
                for k, v in src.attrs.items():
                    dst.attrs[k] = v
                dst.attrs["n_trajectories"] = end - start

                # Group-level attributes (field_names, spatial_dims, bc_type, etc.)
                def copy_group_attrs(name, obj):
                    if isinstance(obj, h5py.Group):
                        grp = dst.require_group(name)
                        for k, v in obj.attrs.items():
                            grp.attrs[k] = v
                src.visititems(copy_group_attrs)

                # Shared datasets (dimensions, boundary_conditions) — copy verbatim
                for name, ds in shared_datasets.items():
                    grp_path = "/".join(name.split("/")[:-1])
                    if grp_path and grp_path not in dst:
                        dst.require_group(grp_path)
                    src.copy(ds, dst, name=name)

                # Trajectory-indexed datasets — slice, preserve chunk layout
                for name, ds in traj_datasets.items():
                    data = ds[start:end]
                    grp_path = "/".join(name.split("/")[:-1])
                    if grp_path and grp_path not in dst:
                        dst.require_group(grp_path)
                    src_chunks = ds.chunks
                    chunks = (min(1, data.shape[0]),) + src_chunks[1:] if src_chunks else None
                    new_ds = dst.create_dataset(name, data=data, chunks=chunks)
                    # Copy dataset-level attributes
                    for k, v in ds.attrs.items():
                        new_ds.attrs[k] = v

            size_mb = out_path.stat().st_size / 1e6
            print(f"    {out_name}: {end - start} traj, {size_mb:.0f} MB")
            created.append(out_path)

    # Delete original only after all splits written successfully
    src_path.unlink()
    print(f"    Removed original: {src_path.name}")
    return created


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert PDEBench SWE to The Well format")
    parser.add_argument("--n_splits", type=int, default=8,
                        help="Number of files to split each split into (default: 8). "
                             "Use 1 to skip splitting.")
    parser.add_argument("--no_split", action="store_true",
                        help="Skip the splitting step entirely (implies --n_splits 1).")
    args = parser.parse_args()

    if args.no_split:
        args.n_splits = 1

    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    src_path = repo_root / PDEBENCH_FILE
    output_base = repo_root / OUTPUT_BASE

    if not src_path.exists():
        print(f"ERROR: Source file not found: {src_path}")
        sys.exit(1)

    print(f"Source: {src_path}")
    print(f"Output: {output_base}")
    print()

    # ── Step 1: Convert ───────────────────────────────────────────
    with h5py.File(src_path, "r") as src:
        n_samples = len(src.keys())
        print(f"PDEBench file has {n_samples} samples")

        meta = read_pdebench_metadata(src)
        print(f"Grid: t={meta['t'].shape}, x={meta['x'].shape}, y={meta['y'].shape}")
        print(f"Constants: gravity={meta['gravity']}, inner_height={meta['inner_height']}")
        print()

        for split, (start, end) in SPLIT_RANGES.items():
            convert_split(src, meta, split, start, end, output_base)

    # ── Step 2: Normalization stats (from training data only) ─────
    compute_stats(output_base)

    # ── Step 3: Split into multiple files ─────────────────────────
    if args.n_splits > 1:
        print(f"\nSplitting each file into {args.n_splits} parts...")
        for split in SPLIT_RANGES:
            single_file = output_base / "data" / split / f"{DATASET_NAME}.hdf5"
            print(f"  {split}/")
            split_file(single_file, args.n_splits)

    print("\nOK: Conversion complete!")
    if args.n_splits > 1:
        print(f"Each split is now stored as {args.n_splits} separate HDF5 files.")
        print("WellDataset automatically discovers all *.hdf5 files in each split directory.")


if __name__ == "__main__":
    main()

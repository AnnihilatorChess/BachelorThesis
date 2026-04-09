"""
Split pdebench_swe.hdf5 (800 trajectories, single file) into N smaller files
so that DataLoader workers can read from them in parallel without HDF5 lock contention.

Usage:
    python split_hdf5.py                  # splits into 8 files (100 traj each) for all splits
    python split_hdf5.py --n_splits 4     # 4 files of 200 traj each
    python split_hdf5.py --dry_run        # print plan without writing
"""

import argparse
import os
import h5py
import numpy as np

SPLITS = ["train", "valid", "test"]
BASE = os.path.dirname(os.path.abspath(__file__))


def split_file(src_path: str, out_dir: str, n_splits: int, dry_run: bool):
    with h5py.File(src_path, "r") as src:
        # Determine n_trajectories first, then classify datasets
        n_traj = src["t0_fields/h"].shape[0]

        # Any dataset whose first dimension equals n_traj is trajectory-indexed
        # (covers both multi-dim fields AND 1D scalars like dam_radius)
        traj_datasets = {}
        shared_datasets = {}

        def collect(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.shape and obj.shape[0] == n_traj:
                traj_datasets[name] = obj.shape
            else:
                shared_datasets[name] = obj.shape

        src.visititems(collect)

        chunk_size = int(np.ceil(n_traj / n_splits))
        print(f"  {os.path.basename(src_path)}: {n_traj} trajectories -> {n_splits} files of ~{chunk_size} each")
        print(f"  Trajectory datasets: {list(traj_datasets.keys())}")
        print(f"  Shared datasets:     {list(shared_datasets.keys())}")

        if dry_run:
            return

        os.makedirs(out_dir, exist_ok=True)

        for i in range(n_splits):
            start = i * chunk_size
            end = min(start + chunk_size, n_traj)
            if start >= n_traj:
                break

            out_name = f"pdebench_swe_{i:02d}.hdf5"
            out_path = os.path.join(out_dir, out_name)

            with h5py.File(out_path, "w") as dst:
                # Copy shared datasets (dimensions, boundary_conditions) verbatim
                for name in shared_datasets:
                    grp = "/".join(name.split("/")[:-1])
                    if grp and grp not in dst:
                        dst.require_group(grp)
                    src.copy(src[name], dst, name=name)

                # Copy trajectory-sliced datasets, preserving original chunk layout
                for name in traj_datasets:
                    data = src[name][start:end]
                    grp = "/".join(name.split("/")[:-1])
                    if grp and grp not in dst:
                        dst.require_group(grp)
                    # Mirror original chunking (first dim = 1 trajectory) so per-sample
                    # access cost is identical to the original file.
                    # No compression: avoids decompression overhead on every sample read.
                    src_chunks = src[name].chunks
                    if src_chunks is not None:
                        # Replace trajectory dim chunk size with 1 (already is 1 in original)
                        new_chunks = (min(1, data.shape[0]),) + src_chunks[1:]
                    else:
                        new_chunks = None
                    dst.create_dataset(name, data=data, chunks=new_chunks)

            size_mb = os.path.getsize(out_path) / 1e6
            print(f"  Written: {out_name} ({end - start} traj, {size_mb:.0f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for split in SPLITS:
        src = os.path.join(BASE, "data", split, "pdebench_swe.hdf5")
        if not os.path.exists(src):
            print(f"Skipping {split}: {src} not found")
            continue
        out_dir = os.path.join(BASE, "data", split)
        print(f"\n=== {split} ===")
        split_file(src, out_dir, args.n_splits, args.dry_run)

    if not args.dry_run:
        print("\nDone. Original files are still present — verify the split files load correctly,")
        print("then delete the originals: rm data/{train,valid,test}/pdebench_swe.hdf5")


if __name__ == "__main__":
    main()

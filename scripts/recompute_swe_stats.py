import h5py
import numpy as np
import yaml
import os
from pathlib import Path

DATASET_NAME = "pdebench_swe"
DATASET_DIR = Path("datasets") / DATASET_NAME
TRAIN_DATA_DIR = DATASET_DIR / "data" / "train"

def compute_and_save_stats():
    print(f"Computing stats for {DATASET_NAME}...")
    
    # Find all hdf5 files in train dir
    train_files = list(TRAIN_DATA_DIR.glob("*.hdf5"))
    if not train_files:
        print(f"No HDF5 files found in {TRAIN_DATA_DIR}")
        return

    all_h = []
    for f_path in train_files:
        print(f"  Reading {f_path.name}...")
        with h5py.File(f_path, "r") as f:
            h_data = f["t0_fields"]["h"][:]
            all_h.append(h_data)
    
    h_data = np.concatenate(all_h, axis=0)
    print(f"  Total samples: {h_data.shape[0]}")

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

    stats_path = DATASET_DIR / "stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

    print(f"DONE: Stats written to {stats_path}")

if __name__ == "__main__":
    compute_and_save_stats()

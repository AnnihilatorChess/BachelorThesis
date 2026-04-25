import h5py
import numpy as np
import yaml
import os
from pathlib import Path

DATASET_NAME = "pdebench_1d_burgers"
DATASET_DIR = Path("datasets") / DATASET_NAME
TRAIN_DATA_DIR = DATASET_DIR / "data" / "train"

def compute_and_save_stats():
    print(f"Computing stats for {DATASET_NAME}...")
    
    # Find all hdf5 files in train dir
    train_files = list(TRAIN_DATA_DIR.glob("*.hdf5"))
    if not train_files:
        print(f"No HDF5 files found in {TRAIN_DATA_DIR}")
        return

    all_u = []
    for f_path in train_files:
        print(f"  Reading {f_path.name}...")
        with h5py.File(f_path, "r") as f:
            u_data = f["t0_fields"]["u"][:]
            all_u.append(u_data)
    
    u_data = np.concatenate(all_u, axis=0)
    print(f"  Total samples: {u_data.shape[0]}")

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

    stats_path = DATASET_DIR / "stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

    print(f"DONE: Stats written to {stats_path}")

if __name__ == "__main__":
    compute_and_save_stats()

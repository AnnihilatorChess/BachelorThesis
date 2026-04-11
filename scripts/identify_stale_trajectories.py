import h5py
import numpy as np
import os
import glob

def identify_stale_trajectories(h5_file, threshold=1e-6):
    """
    Structure: f['t0_fields']['A'] has shape [N_trajs, T, H, W]
    """
    stale_trajectories = []
    with h5py.File(h5_file, 'r') as f:
        # Based on walk_h5.py, the data is in t0_fields/A and t0_fields/B
        if 't0_fields' not in f or 'A' not in f['t0_fields']:
            return []
            
        data_a_ds = f['t0_fields']['A']
        n_trajs = data_a_ds.shape[0]
        
        for i in range(n_trajs):
            # Load one trajectory at a time to be memory efficient
            traj_a = data_a_ds[i] # [T, H, W]
            
            last_steps = traj_a[-100:]
            std_over_time = np.std(last_steps, axis=0).mean()
            
            if std_over_time < threshold:
                diffs = np.diff(traj_a, axis=0)
                mean_abs_diffs = np.abs(diffs).mean(axis=(1, 2))
                
                is_below = mean_abs_diffs < threshold
                flipped_not_below = ~is_below[::-1]
                if not np.any(flipped_not_below):
                    first_steady_step = 0
                else:
                    first_steady_step = len(is_below) - np.argmax(flipped_not_below)
                
                stale_trajectories.append((i, first_steady_step))
                
    return stale_trajectories

def analyze_split(split_path, threshold=1e-6):
    h5_files = glob.glob(os.path.join(split_path, "*.hdf5"))
    split_results = {}
    total_trajs = 0
    total_stale = 0
    
    for f_path in h5_files:
        f_name = os.path.basename(f_path)
        print(f"  Processing {f_name}...")
        try:
            stale = identify_stale_trajectories(f_path, threshold)
            split_results[f_name] = stale
            total_stale += len(stale)
            
            with h5py.File(f_path, 'r') as f:
                if 't0_fields' in f and 'A' in f['t0_fields']:
                    total_trajs += f['t0_fields']['A'].shape[0]
        except Exception as e:
            print(f"    Error processing {f_name}: {e}")
            
    return split_results, total_trajs, total_stale

if __name__ == "__main__":
    base_path = "datasets/gray_scott_reaction_diffusion/data"
    splits = ["test", "valid", "train"]
    threshold = 1e-6
    
    overall_summary = []
    for split in splits:
        print(f"\nAnalyzing split: {split}")
        results, n_total, n_stale = analyze_split(os.path.join(base_path, split), threshold)
        perc = (n_stale / n_total) * 100 if n_total > 0 else 0
        print(f"Split {split}: {n_stale}/{n_total} trajectories are stale ({perc:.2f}%)")
        overall_summary.append((split, n_stale, n_total, perc))
        for f_name, stale_list in results.items():
            if stale_list:
                print(f"    - {f_name}: {len(stale_list)} stale")

    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print("="*40)
    for split, n_stale, n_total, perc in overall_summary:
        print(f"{split:6}: {n_stale:4} / {n_total:4} ({perc:5.2f}%)")

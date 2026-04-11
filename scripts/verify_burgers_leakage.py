import h5py
import numpy as np

def analyze_leakage(filepath):
    print(f"Analyzing {filepath}")
    with h5py.File(filepath, 'r') as f:
        # Get initial conditions for all 10000 trajectories
        # shape is (10000, 201, 1024) -> we want (10000, 1024) at t=0
        u_0 = f['tensor'][:, 0, :]
        
        # Calculate some statistics for each initial condition
        means = np.mean(u_0, axis=1)
        stds = np.std(u_0, axis=1)
        
        # Check if the means or stds are sorted
        is_means_sorted = np.all(np.diff(means) >= 0) or np.all(np.diff(means) <= 0)
        is_stds_sorted = np.all(np.diff(stds) >= 0) or np.all(np.diff(stds) <= 0)
        
        print(f"  Are initial condition means sorted? {is_means_sorted}")
        print(f"  Are initial condition stds sorted? {is_stds_sorted}")
        
        # Check statistics across the splits we use
        splits = {
            "Train (0-8000)": u_0[:8000],
            "Valid (8000-9000)": u_0[8000:9000],
            "Test (9000-10000)": u_0[9000:]
        }
        
        for name, data in splits.items():
            print(f"  {name}: overall mean={np.mean(data):.4f}, overall std={np.std(data):.4f}")

if __name__ == "__main__":
    try:
        analyze_leakage('../../datasets/PDEBench/1D_Burgers/1D_Burgers_Sols_Nu0.01.hdf5')
        print("-" * 40)
        analyze_leakage('../../datasets/PDEBench/1D_Burgers/1D_Burgers_Sols_Nu0.001.hdf5')
    except Exception as e:
        print(f"Error: {e}")

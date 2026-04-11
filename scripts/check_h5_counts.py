import h5py
import glob
import os

for split in ["test", "valid", "train"]:
    print(f"\nSplit: {split}")
    files = sorted(glob.glob(f'datasets/gray_scott_reaction_diffusion/data/{split}/*.hdf5'))
    print(f"{'File':60} | {'Trajs':5}")
    print("-" * 70)
    for f_path in files:
        try:
            f = h5py.File(f_path, 'r')
            tk = [k for k in f.keys() if k.startswith('t') and k.endswith('_fields')]
            print(f"{os.path.basename(f_path):60} | {len(tk):5}")
            f.close()
        except Exception as e:
            print(f"{os.path.basename(f_path):60} | Error: {e}")

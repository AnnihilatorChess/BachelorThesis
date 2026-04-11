import h5py
import os

f_path = 'datasets/gray_scott_reaction_diffusion/data/train/gray_scott_reaction_diffusion_gliders_F_0.014_k_0.054.hdf5'
if os.path.exists(f_path):
    with h5py.File(f_path, 'r') as f:
        print(f"Walking {f_path}...")
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name:40} Shape: {obj.shape}")
            elif isinstance(obj, h5py.Group):
                # Only print groups if they are not trajectory groups (to avoid noise)
                if not (name.startswith('t') and name.endswith('_fields')):
                    print(f"Group:   {name}")
        f.visititems(walk)
else:
    print(f"File not found: {f_path}")

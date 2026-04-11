import h5py
import yaml
import numpy as np

src_path = 'datasets/PDEBench/2D_SWE/2D_rdb_NA_NA.h5'
with h5py.File(src_path, 'r') as f:
    radii = []
    keys = list(f.keys())
    keys.sort()
    print(f"Total keys: {len(keys)}")
    for k in keys:
        config_str = f[k].attrs['config']
        if isinstance(config_str, bytes):
            config_str = config_str.decode('utf-8')
        config = yaml.safe_load(config_str)
        sim = config.get('sim', config)
        radii.append(float(sim['dam_radius']))

radii = np.array(radii)
print(f"First 10 radii: {radii[:10]}")
print(f"Last 10 radii: {radii[-10:]}")
print(f"Is sorted: {np.all(np.diff(radii) >= 0) or np.all(np.diff(radii) <= 0)}")

train_radii = radii[:800]
valid_radii = radii[800:900]
test_radii = radii[900:1000]

print(f"Train: mean={train_radii.mean():.4f}, std={train_radii.std():.4f}, min={train_radii.min():.4f}, max={train_radii.max():.4f}")
print(f"Valid: mean={valid_radii.mean():.4f}, std={valid_radii.std():.4f}, min={valid_radii.min():.4f}, max={valid_radii.max():.4f}")
print(f"Test:  mean={test_radii.mean():.4f}, std={test_radii.std():.4f}, min={test_radii.min():.4f}, max={test_radii.max():.4f}")

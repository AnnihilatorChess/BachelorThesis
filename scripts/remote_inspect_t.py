import h5py
import numpy as np

try:
    f = h5py.File('../../datasets/PDEBench/1D_Burgers/1D_Burgers_Sols_Nu0.01.hdf5', 'r')
    t = f['t-coordinate'][:]
    print(f"t length: {len(t)}")
    print(f"t first 5: {t[:5]}")
    print(f"t last 5: {t[-5:]}")
    
    tensor = f['tensor']
    print(f"tensor shape: {tensor.shape}")
except Exception as e:
    print(e)

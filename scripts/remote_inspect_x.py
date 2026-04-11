import h5py

try:
    f = h5py.File('../../datasets/PDEBench/1D_Burgers/1D_Burgers_Sols_Nu0.01.hdf5', 'r')
    x = f['x-coordinate'][:]
    print(f"x first 5: {x[:5]}")
    print(f"x last 5: {x[-5:]}")
except Exception as e:
    print(e)

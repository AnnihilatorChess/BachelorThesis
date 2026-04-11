import h5py
try:
    f = h5py.File('../../datasets/PDEBench/1D_Burgers/1D_Burgers_Sols_Nu0.01.hdf5', 'r')
    for k in f.keys():
        obj = f[k]
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset {k}: shape {obj.shape}")
        elif isinstance(obj, h5py.Group):
            print(f"Group {k}: num keys {len(obj.keys())}")
            if len(obj.keys()) > 0:
                first_key = list(obj.keys())[0]
                print(f"  First key {first_key} type {type(obj[first_key])}")
                if isinstance(obj[first_key], h5py.Dataset):
                    print(f"  First key shape {obj[first_key].shape}")
except Exception as e:
    print(e)

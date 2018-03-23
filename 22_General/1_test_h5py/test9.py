import h5py
import numpy as np

with h5py.File('myh5py.hdf5','r') as f:
    for key in f['a_group'].keys():
        print(f['a_group'][key].value)

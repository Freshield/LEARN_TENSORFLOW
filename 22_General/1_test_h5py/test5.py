import h5py
import numpy as np

f = h5py.File('myh5py.hdf5','w')

g1 = f.create_group('bar')

g1['dset1'] = np.arange(10)
g1['dset2'] = np.arange(12).reshape((3,4))

for key in g1.keys():
    print(g1[key].name)
    print(g1[key].value)
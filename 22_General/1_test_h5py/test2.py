import h5py
import numpy as np
f = h5py.File('myh5py.hdf5','w')

d1 = f.create_dataset('dset1',(20,),'i')

d1[...] = np.arange(20)

f['dset2'] = np.arange(15)

for key in f.keys():
    print(f[key].name)
    print(f[key].value)
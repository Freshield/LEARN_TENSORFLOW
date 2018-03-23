import h5py
import numpy as np

a = np.arange(200000)

np.save('mynpy.npy',a)

with h5py.File('myh5py.hdf5','w') as f:
    group = f.create_group('a_group')
    group.create_dataset('matrix',data=a,compression='gzip')
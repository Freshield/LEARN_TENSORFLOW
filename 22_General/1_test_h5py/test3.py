import h5py
import numpy as np

f = h5py.File('myh5py.hdf5','w')
a = np.arange(20000)
d1 = f.create_dataset('dset1',data=a)
for key in f.keys():
    print(f[key].name)
    print(f[key].value)

np.save('mynpy.npy',a)
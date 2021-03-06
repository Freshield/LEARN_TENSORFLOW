import h5py
import numpy as np

f = h5py.File('myh5py.hdf5','w')

a = np.arange(20)
d1 = f.create_dataset('dset1',data=a)

d2 = f.create_dataset('dset2',(3,4),'i')
d2[...] = np.arange(12).reshape((3,4))

f['dset3'] = np.arange(15)

for key in f.keys():
    print(f[key].name)
    print(f[key].value)
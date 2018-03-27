import h5py

f = h5py.File('myh5py.hdf5','w')

d1 = f.create_dataset('dset1',(20,),'i')

for key in f.keys():
    print(key)
    print(f[key].name)
    print(f[key].shape)
    print(f[key].value)
import h5py

f = h5py.File('myh5py.hdf5','r')

print(f['/bar1/car1'].keys())
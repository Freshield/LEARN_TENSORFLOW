import h5py
import numpy as np

f = h5py.File('myh5py.hdf5','w')

g1 = f.create_group('bar1')
g2 = f.create_group('bar2')
d = f.create_dataset('dset',data=np.arange(10))

c1 = g1.create_group('car1')
d1 = g1.create_dataset('dset1',data=np.arange(10))

c2 = g2.create_group('car2')
d2 = g2.create_dataset('dset2',data=np.arange(10))

print('...................')
for key in f.keys():
    print(f[key].name)


print('###################')
for key in g1.keys():
    print(g1[key].name)


print('!!!!!!!!!!!!!!!!!!')
for key in g2.keys():
    print(g2[key].name)

print('@@@@@@@@@@@@@@@@@@@@')
print(c1.keys())
print(c2.keys())
import numpy as np

np_data = np.arange(82).reshape([2,41])

a = np.tile(np_data[:,:41],2)

print a
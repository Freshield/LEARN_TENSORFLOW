import numpy as np
import pandas as pd

a = np.arange(12, dtype=float).reshape([3,4])

print a

b = np.max(a[:,:3])
c = np.min(a[:,:3])

print b
print c

a[:,:3] = (a[:,:3] - c) / (b - c)

print a

filename = 'pca1000_1.csv'
dataset = pd.read_csv(filename, header=None)

raw_data = dataset.values

def norm_dataset(dataset, start, stop):
    max = np.max(dataset[:,start:stop])
    min = np.min(dataset[:,start:stop])
    normed = (dataset[:,start:stop] - min) / (max - min)
    return normed

raw_data[:,0:200] = norm_dataset(raw_data, 0, 200)
raw_data[:,200:201] = norm_dataset(raw_data, 200, 201)
raw_data[:,201:221] = norm_dataset(raw_data, 201, 221)
raw_data[:,221:241] = norm_dataset(raw_data, 221, 241)

np.savetxt('norm.csv', raw_data, delimiter=',')

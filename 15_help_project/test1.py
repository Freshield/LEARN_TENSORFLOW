import pandas as pd
import numpy as np

data = pd.read_csv('B.csv', header=None)

print data.shape

a = np.arange(5).reshape((5,1))

print a

b = np.concatenate([a,a], axis=1)
print b
print b.shape

mu = np.arange(5).reshape((5,1))

print mu

mu_array = np.zeros((5,4))

print mu_array

for i in range(4):
    mu_array[:,i] = np.reshape(mu,(5))

print mu_array



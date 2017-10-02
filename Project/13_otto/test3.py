import pandas as pd
import numpy as np

minmax_array = pd.read_csv('data/min_max.csv', header=None, dtype=np.float32).values

# get min and max
min_values = minmax_array[0]
max_values = minmax_array[1]

print min_values
print max_values

a = np.arange(20).reshape(4,5)
print a

x,y = a.shape
print x
print y

b = np.zeros((x,y-1))
print b
c = np.zeros((x,y-2))
print c

c[:,:] = a[:,1:-1]
print c

c = c - 10
print c
b[:,-1] = a[:,-1]
print b
b[:,:-1] = c[:,:]
print b
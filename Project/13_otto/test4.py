import pandas as pd
import numpy as np

data_size = 10

indexs = np.arange(data_size)
np.random.shuffle(indexs)

print indexs

a = np.arange(20).reshape(10,2)
print a

print

b = a[indexs[:5]]
print b

data = pd.read_csv('data/test.csv')
print data.shape
print data.values.shape
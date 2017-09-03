import numpy as np
import pandas as pd
"""
totoal_data = np.zeros((10,10))
for i in range(10):
    data = np.zeros((10)) + i
    print data
    totoal_data[i] = data
np.savetxt('data/num_data.csv', totoal_data,delimiter=',')
"""

filename = 'data/num_data.csv'
reader = pd.read_csv(filename,header=None,chunksize=1)

for chunk in reader:
    print chunk.values.shape
import numpy as np
import pandas as pd

for j in range(10):
    totoal_data = np.zeros((100, 10))

    for i in range(100):
        data = np.zeros((10)) + i + 100 * j
        print data
        totoal_data[i, :] = data
    np.savetxt('data/10files/%d_data.csv' % j, totoal_data, delimiter=',')
"""

filename = 'data/1000num_data.csv'
reader = pd.read_csv(filename,header=None,chunksize=1)

for chunk in reader:
    print chunk.values.shape
"""
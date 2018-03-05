import pandas as pd
import numpy as np
"""
filename = '/media/freshield/New_2T_Data/Ciena/new_data/FiberID_6fibers_20Spans/FiberID_6fibers_20Spans_noPCA_1.csv'

data = pd.read_csv(filename, header=None, nrows=100)

chunk = data.values

np.savetxt('data_sample.csv', chunk, delimiter=",")
"""
filename = 'data_sample.csv'

data = pd.read_csv(filename, header=None)

print data.shape
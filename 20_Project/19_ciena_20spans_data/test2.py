import pandas as pd
import numpy as np

filename = '/media/freshield/DATA/Ciena_new_data/20spans/norm/Raw_data_0_test.csv'

data = pd.read_csv(filename,header=None).values

print data.shape

print np.min(data[:,24041:24061])
print np.max(data[:,24041:24061])
print np.mean(data[:,24041:24061])
import pandas as pd
import numpy as np

filename = 'sample/sample_set.csv'

data = pd.read_csv(filename,header=None).values

print data.shape

print np.min(data[:,24061:24081])
print np.max(data[:,24061:24081])
print np.mean(data[:,24061:24081])
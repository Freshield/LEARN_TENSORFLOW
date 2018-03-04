import pandas as pd
import numpy as np

filename = 'enlc_diff.csv'

data = pd.read_csv(filename,header=None).values

print np.max(data)
print np.min(data)
print np.mean(data)
print np.percentile(abs(data),91)
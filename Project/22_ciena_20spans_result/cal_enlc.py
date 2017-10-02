import pandas as pd
import numpy as np

filename = 'result_probability.csv'

data = pd.read_csv(filename, header=None).values

ENLC_array = np.array([34.515,23.92,21.591,25.829,28.012,29.765]).reshape((6,1))

result = np.matmul(data,ENLC_array)

print result.shape
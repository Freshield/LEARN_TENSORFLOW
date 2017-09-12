import numpy as np
import pandas as pd

train_filename = 'data/norm/train_set.csv'
f_test_filename = 'data/test.csv'

train_data = pd.read_csv(train_filename, header=None)
test_data = pd.read_csv(f_test_filename)

print train_data.values.shape
print test_data.values.shape

train_set_value = train_data.values[:,:].astype(np.float32)
test_set_value = test_data.values[:,1:].astype(np.float32)

print test_set_value[0]
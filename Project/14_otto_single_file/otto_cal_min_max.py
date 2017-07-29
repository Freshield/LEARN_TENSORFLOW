import pandas as pd
import numpy as np

filename = 'data/train.csv'

data = pd.read_csv(filename)

print data.shape

test_size = 878

indexs = np.arange(len(data))
np.random.shuffle(indexs)

train_set = data.values[indexs[0:-878]]
test_set = data.values[indexs[-878:]]

print train_set.shape
print test_set.shape
print indexs[-10:]

min_value = np.min()
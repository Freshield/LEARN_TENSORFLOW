import numpy as np
import pandas as pd

train_filename = 'data/train.csv'
test_filename = 'data/test.csv'

train_data = pd.read_csv(train_filename)
test_data = pd.read_csv(test_filename)

print train_data.values.shape
print test_data.values.shape


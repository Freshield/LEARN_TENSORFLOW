import pandas as pd
import time
from CNN_model import *

NROWS = 100000

SPAN=[10]

filename = '/media/freshield/LINUX/Ciena/CIENA/raw/FiberID_Data_noPCA.csv'

data = pd.read_csv(filename, header=None, nrows=NROWS)

print 'Data Shape: %s' % str(data.shape)

train_set, validation_set, test_set = split_dataset(data, radio=0.1)

#normalize dataset
train_set, train_min, train_max = normalize_dataset(train_set)
validation_set, _, _ = normalize_dataset(validation_set, train_min, train_max)
test_set, _, _ = normalize_dataset(test_set, train_min, train_max)

#reshape dataset
X_train, y_train = reshape_dataset(train_set, SPAN)
X_valid, y_valid = reshape_dataset(validation_set, SPAN)
X_test, y_test = reshape_dataset(test_set, SPAN)

#hypers
reg = 1e-4
lr_rate = 0.002
max_step = 30000
batch_size = 100
lr_decay = 0.99
lr_epoch = 800

log = ''

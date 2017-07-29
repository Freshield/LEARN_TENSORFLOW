import pandas as pd
import numpy as np

filename = 'data/train.csv'

dic = {'Class_1':0, 'Class_2':1, 'Class_3':2, 'Class_4':3, 'Class_5':4, 'Class_6':5, 'Class_7':6, 'Class_8':7, 'Class_9':8}


def calcul_norm_new(dataset, min, max, mean):
    return (2 * dataset - 2 * mean) / (max - min)

data = pd.read_csv(filename)

print data.shape

test_size = 878

indexs = np.arange(len(data))
np.random.shuffle(indexs)

train_set = data.values[indexs[0:-878],1:]
test_set = data.values[indexs[-878:],1:]

train_set_value = data.values[indexs[0:-878],1:-1].astype(np.float32)
test_set_value = data.values[indexs[-878:],1:-1].astype(np.float32)

print train_set_value.shape
print test_set_value.shape
print indexs[-10:]

min_value = np.min(train_set_value, axis=0)
max_value = np.max(train_set_value, axis=0)
mean_value = np.mean(train_set_value, axis=0)

print max_value

norm_train_data = np.zeros((61000, 94))
norm_test_data = np.zeros((878,94))

norm_train_data[:,:-1] = calcul_norm_new(train_set_value, min_value, max_value, mean_value)
norm_test_data[:,:-1] = calcul_norm_new(test_set_value, min_value, max_value, mean_value)


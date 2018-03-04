import pandas as pd
import numpy as np

train_filename = 'data/train.csv'
test_filename = 'data/test.csv'

dic = {'Class_1':0., 'Class_2':1., 'Class_3':2., 'Class_4':3., 'Class_5':4., 'Class_6':5., 'Class_7':6., 'Class_8':7., 'Class_9':8.}


def calcul_norm_new(dataset, min, max, mean):
    return (2 * dataset - 2 * mean) / (max - min)

train_data = pd.read_csv(train_filename)
test_data = pd.read_csv(test_filename)

train_size = 61878
test_size = 144368

#get the 93 features coloums
train_set = train_data.values[:,1:-1].astype(np.float32)
test_set = test_data.values[:,1:].astype(np.float32)

#get min max mean values
min_value = np.min(train_set, axis=0)
max_value = np.max(train_set, axis=0)
mean_value = np.mean(train_set, axis=0)

norm_test_data = np.zeros((test_size,93))

norm_test_data[:,:] = calcul_norm_new(test_set, min_value, max_value, mean_value)

np.savetxt('data/norm/testing.csv', norm_test_data, delimiter=',')
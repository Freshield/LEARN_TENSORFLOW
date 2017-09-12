import pandas as pd
import numpy as np

train_filename = 'data/norm/train_set.csv'
f_test_filename = 'data/test.csv'

dic = {'Class_1':0., 'Class_2':1., 'Class_3':2., 'Class_4':3., 'Class_5':4., 'Class_6':5., 'Class_7':6., 'Class_8':7., 'Class_9':8.}


def calcul_norm_new(dataset, min, max, mean):
    return (2 * dataset - 2 * mean) / (max - min)

#read train and final test file
train_data = pd.read_csv(train_filename,header=None)
test_data = pd.read_csv(f_test_filename)

test_size = 144368

#the last value is category
train_set_value = train_data.values[:,:-1].astype(np.float32)
test_set_value = test_data.values[:,1:].astype(np.float32)

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

for i in xrange(norm_train_data.shape[0]):
    norm_train_data[i,-1] = dic[train_set[i,-1]]


for i in xrange(norm_test_data.shape[0]):
    norm_test_data[i,-1] = dic[test_set[i,-1]]

#np.savetxt('data/norm/train_set.csv', norm_train_data, delimiter=',')
#np.savetxt('data/norm/test_set.csv', norm_test_data, delimiter=',')

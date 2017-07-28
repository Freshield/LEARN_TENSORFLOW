import pandas as pd
import numpy as np

filename = 'data/sample.csv'

data = pd.read_csv(filename)

print data

value = data.values[:,1:]
print value

dic = {'Class_1':1, 'Class_2':2, 'Class_3':3}


for i in range(len(value)):
    value[i,-1] = dic[value[i,-1]]

print value

result = value.astype(np.float32)

print result

data2 = pd.read_csv('data/train.csv')

print data2.shape
value2 = data2.values[1:,1:]

print value2.shape
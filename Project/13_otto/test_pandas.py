import pandas as pd
import numpy as np

filename = 'data/sample.csv'

data = pd.read_csv(filename, skiprows=0)

print data

value = data.values[:,1:]
print value

dic = {'Class_1':1, 'Class_2':2, 'Class_3':3}

"""
for i in range(len(value)):
    value[i,-1] = dic[value[i,-1]]
"""
print value

result = np.zeros((4,6))

print result

result[:,:-1] = value[:,:-1]

print result
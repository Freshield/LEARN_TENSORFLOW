import numpy as np
import pandas as pd

filename = 'data/clean_raw_train_35.csv'

data = pd.read_csv(filename).values

print data.shape

#a = np.arange(8).reshape((2,4))
#print np.min(data,axis=0)
#print np.max(data,axis=0)


label_ones = np.zeros((590,36))
count = 0
for i in xrange(24000):
    if data[i,-1] == 1:
        label_ones[count,:] = data[i,:]
        count += 1

np.savetxt('data/train_label_one_set.csv', label_ones, delimiter=',')
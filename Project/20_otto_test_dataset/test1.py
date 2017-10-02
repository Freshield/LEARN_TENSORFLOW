import pandas as pd
import numpy as np


def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    #index = tf.random_shuffle(tf.range(0, data_size))#maybe can do it on tensorflow later
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs

data = pd.read_csv('data/norm/train_set.csv', header=None)

index = get_random_seq_indexs(data)
print index[:10]

#print data.values[[0,1,2,3,4],-1]
batch_data = data.values[index[:100]]
print batch_data.shape
print batch_data[:10,-1]
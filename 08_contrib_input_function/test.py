import tensorflow as tf
import numpy as np
import pandas as pd
import itertools

feature_column = np.array([1,2.4,0,9.9,3,120])

feature_tensor = tf.constant(feature_column)

sparse_tensor = tf.SparseTensor(indices=[[0,1],[2,4]],
                                values=[6, 0.5],
                                dense_shape=[3,5])

print (sparse_tensor.values)
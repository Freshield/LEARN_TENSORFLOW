import tensorflow as tf
import numpy as np

sparse_tensor = tf.SparseTensor(indices=[[0,1], [2,4]],
                                values=[6, 0.5],
                                dense_shape=[3, 5])

sess = tf.InteractiveSession()

print (sparse_tensor.eval()).values
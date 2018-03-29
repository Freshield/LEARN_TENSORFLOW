import tensorflow as tf
import numpy as np

a = tf.constant(np.zeros((1,2,3)))
b = tf.constant(np.zeros((1,2,3)))

a = tf.reshape(a,(1,2,3,1))
b = tf.reshape(b,(1,2,3,1))
print(a.shape)

list = []
list.append(a)
list.append(b)

c = tf.concat(list,axis=3)
print(c.shape)
import tensorflow as tf
import numpy as np
import pandas as pd

filename = 'data/result_trapha_T.csv'

data = pd.read_csv(filename,header=None)

print data.values[:10]

input_data = tf.placeholder(tf.float32, [None,9])

norm = tf.nn.softmax(input_data)

the_argmax = tf.argmax(norm, axis=1)

one_hot = tf.one_hot(the_argmax, depth=9, dtype=tf.int32)

sess = tf.InteractiveSession()

feed_dict = {input_data:data.values[:10,1:]}

res, argmax_v, one_hot_v  = sess.run([norm, the_argmax, one_hot], feed_dict=feed_dict)

print res

print argmax_v

print one_hot_v



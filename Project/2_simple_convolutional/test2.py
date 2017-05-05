import tensorflow as tf
import numpy as np
import pandas as pd

ta = tf.ones([3])

tb = tf.zeros([2])

print ta
print tb

tc = tf.concat([ta, tb], axis=0)
print tc

a = np.arange(10)

print a

print a[5:]

filename = '/home/freshield/ciena_test/ciena10000.csv'

dataset = pd.read_csv(filename, header=None)

print dataset.shape

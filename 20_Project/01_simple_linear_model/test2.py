import numpy as np
import tensorflow as tf
import pandas as pd

sess = tf.InteractiveSession()

a = np.arange(10)

print a

np.random.shuffle(a)

print a

ta = tf.range(0, 10)

tb = tf.random_shuffle(ta)

print ta.eval()

print tb.eval()

dataset = pd.read_csv('ciena_test.csv', header=None)

print dataset.shape

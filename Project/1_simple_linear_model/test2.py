import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

a = np.arange(10)

print a

np.random.shuffle(a)

print a

ta = tf.range(0, 10)

tb = tf.random_shuffle(ta)

print ta.eval()

print tb.eval()
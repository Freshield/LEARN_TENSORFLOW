import tensorflow as tf
import time

time_before = time.time()
a = tf.constant('hello\nworld', shape=[10000])

b = tf.string_split(a, '\n')

sess = tf.InteractiveSession()

c = b.eval().values

time_after = time.time()

print a.eval()
print c
print c.shape
print time_after - time_before
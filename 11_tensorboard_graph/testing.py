import tensorflow as tf
import time

time_before = time.time()
a = tf.constant('hello\nworld', shape=[5])

b = tf.string_split(a, '\n')

sess = tf.Session()
"""
c = b.eval().values

time_after = time.time()

print a.eval()
print c
print c.shape
print time_after - time_before
time_before = time.time()


with tf.device('/gpu:0'):
    file = tf.gfile.Open('facebook_file')

    text = file.read()

    d = tf.constant([text])

    e = tf.string_split(d, '\n')

    #f = tf.string_split(e.values, '\t')

    #g = e.shape
#print d.eval().shape

sess.run(d)
time_after = time.time()
#print e.eval().values
#print e.eval().values.shape
#print g
#print g.shape
print time_after - time_before
"""

import tensorflow as tf
sess = tf.InteractiveSession()

with tf.device('/gpu:0'):
    text = tf.constant("aa a", name="LEFT")
    result = tf.string_split([text], delimiter=" ")
print result.eval()
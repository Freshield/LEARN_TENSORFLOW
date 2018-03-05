import numpy as np
import tensorflow as tf

a = np.arange(24)
print a

np_reshape = a.reshape((2,3,4))
print np_reshape
print a.reshape((3,4,2))


with tf.Graph().as_default():
    with tf.Session() as sess:
        tf_raw = tf.placeholder(tf.float32,[24])
        tf_reshape = tf.reshape(tf_raw,(2,3,4))

        feed_dict = {tf_raw:a}

        result = sess.run(tf_reshape,feed_dict=feed_dict)

        print result


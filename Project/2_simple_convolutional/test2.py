import tensorflow as tf

ta = tf.ones([3])

tb = tf.zeros([2])

print ta
print tb

tc = tf.concat([ta, tb], axis=0)
print tc

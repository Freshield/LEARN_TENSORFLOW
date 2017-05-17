import tensorflow as tf

a = tf.ones([2,3],tf.float32)

b = tf.pad(a,[[1,1],[2,2]])

print a
print b
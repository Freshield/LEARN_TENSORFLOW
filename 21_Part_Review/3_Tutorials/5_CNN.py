import tensorflow as tf

a = tf.constant([1,2,3])

sess = tf.InteractiveSession()

print(a.eval())

print(tf.__version__)
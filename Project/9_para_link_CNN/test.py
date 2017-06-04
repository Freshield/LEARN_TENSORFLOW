import tensorflow as tf

a = tf.zeros([10,5])
b = tf.ones([10,2])

c = tf.concat([a,b], axis=1)

sess = tf.InteractiveSession()

print c.eval()
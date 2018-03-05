import tensorflow as tf

x = tf.reshape(tf.range(10,dtype=tf.float32), [2,5])

y = tf.argmax(x, 1)

y_prob = tf.nn.softmax(x)

ENLC_array = tf.reshape(tf.constant([34.515, 23.92, 21.591, 25.829, 28.012, 29.765], dtype=tf.float32),
                                    [6,1])

sess = tf.InteractiveSession()

print x.eval()
print y.eval()
print y_prob.eval()
print ENLC_array.eval()
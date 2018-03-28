import tensorflow as tf
import numpy as np

a = np.arange(12).reshape((3,4))
b = tf.Variable(a,name='b')

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

print(b.eval())
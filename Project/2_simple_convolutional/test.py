import numpy as np
import tensorflow as tf

padding = np.zeros([4, 1, 3])
mid = np.ones([4, 2, 3])

print np.concatenate((padding, mid, padding), axis=1)

x = tf.constant(np.arange(24))
x_image = tf.reshape(x, [-1, 2, 3])

x_image = tf.pad(x_image, [[0, 0], [1, 1], [0, 0]], 'CONSTANT')



sess = tf.InteractiveSession()

print x_image.eval()

a = np.arange(24).reshape([8, 3])
print a
b = np.array([0, 1, 2])
print a[b]

y = tf.constant(np.array([[0,1,2],[0,0,0],[1,1,1]]))
print y.eval()
y_one_hot = tf.one_hot(y, 3)
print y_one_hot.eval()



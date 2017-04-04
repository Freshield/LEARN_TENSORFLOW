import numpy as np
import tensorflow as tf

x = np.eye(3)

print x

y = np.nonzero(x)

print y[0],y[1]

print x.shape

print x[::1]

x = [x for x in range(100)]

print x

y = x[0:100:10]

print y

y = x[::10]
print y

x = np.eye(3)

y = x[::1]

print y

y = x

print y

y[0] = 10

print y
print x

y[0] = [1,0,0]

x[0][2] = 1
print x

y = np.nonzero(x)
z = y[1]

print y
print x
print z
print z.shape

print x

y = tf.stack(x)
print y
z = tf.constant(x)
print z
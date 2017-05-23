import numpy as np
import tensorflow as tf
import time
"""


x = np.eye(3000)

print x

y = np.nonzero(x)

print y[0],y[1]

print x.shape

print x[::1]

x = [x for x in range(3000)]

print x

y = x[0:100:10]

print y

y = x[::10]
print y

x = np.eye(5000)

y = x[::1]

print y

y = x

print y

y[0] = 10

print y
print x

y[0] = [0]
y[0][0] = 1

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
w = tf.Variable(y, name='w')
print w
z = tf.constant(x)
print z
v = tf.Variable(z, name='v')
print v
"""
sess = tf.InteractiveSession()
time_before = time.time()
x = tf.ones([10000,10000])
a = tf.map_fn(lambda z: z + 100, x)
b = tf.reduce_mean(a)
time_after = time.time()
diff_time = time_after -time_before

print a.eval()
print b.eval()
print diff_time

time_before = time.time()
#9 is \t, 10 is \n
test = "1475349108	72	8cf0bcb30983a0f6	fedef371bab5ae04	b1bde6ff	93c9e166	6	d9ac464b	3c52b4d4	e43e8c29	521568b2	c2c389c8	926920bb	1	1" \
       "\n1475349108	1500	8cf0bcb30983a0f6	19822f60db7c89e3	017af400	2d499bef	6	d9ac464b	3c52b4d4	e43e8c29	aa2e5ade	c2c389c8	926920bb	1	1"
ascii = tf.stack(map(ord, test))
time_after = time.time()
diff_time = time_after -time_before
print ascii.eval()
print diff_time


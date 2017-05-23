import tensorflow as tf


W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

liner_module = W * x + b

sess = tf.Session()

init = tf.global_variables_initializer()

y = tf.placeholder(tf.float32)

squared_deltas = tf.square(liner_module - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)

for i in range(2000):
    print i
    sess.run(train, {x: [1,2,3,4], y:[0,-1,-2,-3]})

print (sess.run([W, b]))
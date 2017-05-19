import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_module = x * W + b
y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(linear_module - y))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

sess = tf.Session()
sess.run(init)

for i in xrange(5000):
    print i
    sess.run(optimizer, {x: x_train, y: y_train})

cur_W, cur_b, cur_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print ("W: %s b: %s loss: %s" % (cur_W, cur_b, cur_loss))

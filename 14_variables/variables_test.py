import tensorflow as tf

weights = tf.Variable(tf.random_normal([784,200], stddev=0.35), name='weights')

biases = tf.Variable(tf.zeros([200]), name='biases')

with tf.device('/cpu:0'):
    v = tf.Variable(tf.zeros([200]))

with tf.device('/gpu:0'):
    g = tf.Variable(tf.zeros([200]))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(weights)
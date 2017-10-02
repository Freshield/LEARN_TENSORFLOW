import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=1.0))
b1 = tf.Variable(tf.zeros([300]))
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
keep_prob = tf.placeholder(tf.float32)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob=keep_prob)


W2 = tf.Variable(tf.truncated_normal([300, 10], stddev=1.0))
b2 = tf.Variable(tf.zeros([10]))
y = tf.matmul(hidden1_drop, W2) + b2

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in xrange(20000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})

    if (i % 100 == 0):
        print ('the %d loss is: %f' % (i, loss))
        result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})
        print ("the %d accuracy is: %f" % (i, result))
        print '---------------------------------------'


result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})
print ("final accuracy is: %f" % (result))

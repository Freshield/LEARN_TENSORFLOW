from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None,784])
y_ = tf.placeholder(tf.float32, shape=[None, 10,])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch[0],y_:batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy = tf.reduce_mean((tf.cast(correct_prediction,tf.float32)))

print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
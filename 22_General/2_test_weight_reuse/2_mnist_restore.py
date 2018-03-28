from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets('../../data/mnist', one_hot=True)

save_path = '../../data/mnist/model/'
if not os.path.exists(save_path):
    tf.gfile.MakeDirs(save_path)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]),name='W')
b = tf.Variable(tf.zeros([10]),name='b')
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

saver = tf.train.Saver()

saver.restore(sess,save_path+'model.ckpt')
print('done restore')

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))


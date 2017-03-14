import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
data_sets = input_data.read_data_sets('MNIST_data', one_hot=True)

print data_sets.train.images.shape
print data_sets.validation.images.shape
print data_sets.test.images.shape
print data_sets.train.num_examples

images = tf.placeholder(tf.float32, shape=(100, 28 * 28))

images, _ = data_sets.train.next_batch(100, False)

print images.shape

print data_sets.train.num_examples

print 10 / 3
print 10 // 3
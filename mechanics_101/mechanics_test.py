import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
data_sets = input_data.read_data_sets('MNIST_data', one_hot=True)

print data_sets.train.images.shape
print data_sets.validation.images.shape
print data_sets.test.images.shape

images_placeholder = tf.placeholder(tf.float32, shape=)
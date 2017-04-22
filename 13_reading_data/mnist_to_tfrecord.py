import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

data_sets = mnist.read_data_sets('tmp/mnist',
                                 dtype=tf.uint8,
                                 reshape=False,
                                 validation_size=5000)

print data_sets.train.images.shape
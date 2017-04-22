import tensorflow as tf
import numpy as np
import pandas as pd

sess = tf.InteractiveSession()

filename = 'iris_test.csv'

pd_file = pd.read_csv('ciena.csv')

print pd_file.shape

filename_queue = tf.train.string_input_producer([filename])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[1.] for x in xrange(5)]
columns = tf.decode_csv(
    value, record_defaults=record_defaults
)

features = columns[0:4]

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)



list = np.ones([4,4])

for i in xrange(8):
    for i in xrange(4):
        example_otehrs = sess.run(features)
        list[i] = example_otehrs

batch = tf.constant(list)

coord.request_stop()
coord.join(threads)

print list

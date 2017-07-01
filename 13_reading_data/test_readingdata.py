import tensorflow as tf
import numpy as np
import pandas as pd

sess = tf.InteractiveSession()

filename = 'iris_test.csv'

pd_file = pd.read_csv('ciena.csv')

print pd_file.shape

filename_queue = tf.train.string_input_producer(['ciena.csv'])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[1.] for x in xrange(6261)]
columns = tf.decode_csv(
    value, record_defaults=record_defaults
)

real_C_matrix = tf.stack(columns[0:3100])
imaginary_C_matrix = tf.stack(columns[3100:6200])
others = tf.stack(columns[6200:6261])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)



list = np.ones([5,61])

for i in xrange(5):
    example_otehrs = sess.run(others)
    list[i] = example_otehrs

batch = tf.constant(list)

coord.request_stop()
coord.join(threads)

print list[4]
print batch.eval().shape

for i in xrange(5):
    example_otehrs = sess.run(others)
    list[i] = example_otehrs


for i in xrange(5):
    example_otehrs = sess.run(others)
    list[i] = example_otehrs


for i in xrange(5):
    example_otehrs = sess.run(others)
    list[i] = example_otehrs

print list[4]

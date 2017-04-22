import tensorflow as tf
import numpy as np
import pandas as pd

sess = tf.InteractiveSession()

filename = 'iris_test.csv'

pd_file = pd.read_csv('ciena.csv')

print pd_file.shape



def read_file(filename_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[1.] for x in xrange(5)]
    columns = tf.decode_csv(
        value, record_defaults=record_defaults
    )
    features = columns[0:4]
    labels = columns[4]
    return features, labels

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer([filenames])
    features, label = read_file(filename_queue)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    features_batch, label_batch = tf.train.shuffle_batch(
        [features, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    return features_batch, label_batch

features, label = input_pipeline(filename, 5)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print features.eval()
print features.eval()
print features.eval()
print features.eval()
print features.eval()

coord.request_stop()
coord.join(threads)


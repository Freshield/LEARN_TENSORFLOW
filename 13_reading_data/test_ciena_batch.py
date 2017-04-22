import tensorflow as tf
import numpy as np
import pandas as pd

sess = tf.InteractiveSession()

filename = 'ciena.csv'

#pd_file = pd.read_csv('ciena.csv')

#print pd_file.shape



def read_file(filename_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[1.] for x in xrange(6261)]
    columns = tf.decode_csv(
        value, record_defaults=record_defaults
    )

    real_C_matrix = tf.stack(columns[0:3100])
    imaginary_C_matrix = tf.stack(columns[3100:6200])
    others = tf.stack(columns[6200:6261])
    return real_C_matrix, imaginary_C_matrix, others

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer([filenames])
    real_C, img_C, others = read_file(filename_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    real_C_batch, img_C_batch, others_batch = tf.train.shuffle_batch(
        [real_C, img_C, others], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    return real_C_batch, img_C_batch, others_batch

real_C_batch, img_C_batch, others_batch = input_pipeline(filename, 5)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print others_batch.eval()

coord.request_stop()
coord.join(threads)


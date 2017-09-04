import tensorflow as tf
import pandas as pd

filename = '/media/freshield/New_2T_Data/Ciena_new_data/20spans/tfrecords_norm/Raw_data_0_train.tfrecords'
#filename = 'data/test_fuc/sample_set.tfrecords'
count = 0
try:
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        labels = example.features.feature['labels'].float_list.value

        print labels[-1]
        count += 1
        print count
except :
    print 'Some error in read data'

print count

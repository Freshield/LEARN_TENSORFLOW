import tensorflow as tf
import pandas as pd
import numpy as np

def csv_to_tfrecord(filename, writer):
    # create writer

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    reader = pd.read_csv(filename, header=None, chunksize=1)

    for line in reader:
        data = line.values.reshape((10))
        print data
        lable = int(data[-1])
        print lable
        features = data[:-1]
        example = tf.train.Example(features=tf.train.Features(
            feature={'lable': _int64_feature([lable]),
                'features': _float_feature(features)}))
        writer.write(example.SerializeToString())

    writer.close()
    print 'Successfully convert'

filename = 'data/num_data.csv'
writer = tf.python_io.TFRecordWriter('data/data.tfrecords')
csv_to_tfrecord(filename,writer)
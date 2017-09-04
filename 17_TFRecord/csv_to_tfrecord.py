import tensorflow as tf
import pandas as pd
import numpy as np

def csv_to_tfrecord(filename, writer):

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    # create writer

    reader = pd.read_csv(filename, header=None, chunksize=1)

    count = 0

    for line in reader:
        print count
        data = line.values.reshape((24081))

        real_C = data[0:12000]
        imag_C = data[12000:24000]
        netCD = data[24000:24001]
        length = data[24001:24021]
        power = data[24021:24041]
        ENLC = data[24041:24061]
        labels = data[24061:24081]

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'real_C': _float_feature(real_C),
                'imag_C': _float_feature(imag_C),
                'netCD' : _float_feature(netCD),
                'length': _float_feature(length),
                'power' : _float_feature(power),
                'ENLC'  : _float_feature(ENLC),
                'labels': _float_feature(labels)
            }))
        writer.write(example.SerializeToString())
        count += 1

    writer.close()
    print 'Successfully convert'

dir_name = ''
filename = '/media/freshield/CORSAIR/LEARN_TENSORFLOW/Project/19_ciena_20spans_data/sample/sample_set.csv'
writer = tf.python_io.TFRecordWriter('data/sample.tfrecords')

csv_to_tfrecord(filename, writer)

writer.close()

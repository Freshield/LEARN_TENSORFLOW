import tensorflow as tf
import pandas as pd

filename = 'data/sample.tfrecords'

for serialized_example in tf.python_io.tf_record_iterator(filename):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    real_C = example.features.feature['real_C'].float_list.value
    labels = example.features.feature['labels'].float_list.value

    print real_C
    print labels


    reader = pd.read_csv('/media/freshield/CORSAIR/LEARN_TENSORFLOW/Project/19_ciena_20spans_data/sample/sample_set.csv', header=None, chunksize=1)

    for line in reader:
        data = line.values.reshape((24081))
        print data[0:12000]
        print data[24061:24081]
        break


    break
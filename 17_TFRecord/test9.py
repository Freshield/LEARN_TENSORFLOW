import tensorflow as tf

for serialized_example in tf.python_io.tf_record_iterator('data/test_fuc/sample_set.tfrecords'):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    length = example.features.feature['length'].float_list.value
    labels = example.features.feature['labels'].float_list.value

    print length
    print labels
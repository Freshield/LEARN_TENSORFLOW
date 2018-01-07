import tensorflow as tf

for serialized_example in tf.python_io.tf_record_iterator('data/data.tfrecords'):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    features = example.features.feature['features'].float_list.value
    lable = example.features.feature['lable'].int64_list.value

    print(features)
    print(lable[0])
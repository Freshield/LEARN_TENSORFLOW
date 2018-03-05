import tensorflow as tf

def read_my_file_format(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={'lable': tf.FixedLenFeature([], tf.int64),
        'features': tf.FixedLenFeature([9], tf.float32)})

    input_data = features['features']
    # input_data = tf.reshape(input_data,[2,5])
    lable_data = features['lable']
    return input_data, lable_data

filenames = ['data/10files/%d_data.tfrecords' % i for i in range(10)]

print filenames

def input_pipeline(filenames, batch_size, read_threads, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

    reader = tf.TFRecordReader()
    example_list = [read_my_file_format(filename_queue) for _ in range(read_threads)]

    batch_size = batch_size
    min_after_dequeue = 2 * batch_size
    capacity = min_after_dequeue + 3 * batch_size

    input_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return input_batch, label_batch

print 'here'

input_batch, label_batch = input_pipeline(filenames, 10, 5)

init = tf.global_variables_initializer()

total_list = []

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    try:
        while not coord.should_stop():
            for i in range(125):
                val, l = sess.run([input_batch, label_batch])
                total_list.extend(l)
                print val
                print l
            coord.request_stop()
    except tf.errors.OutOfRangeError:
        print ('Done training')
    finally:
        coord.request_stop()

coord.join(threads)
sess.close()

total_set = set(total_list)
print len(total_list)
print len(total_set)
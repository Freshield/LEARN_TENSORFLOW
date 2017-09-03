import tensorflow as tf

filename = 'data/data.tfrecords'

filename_queue = tf.train.string_input_producer([filename])

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(serialized_example,
                                   features={
                                       'lable':tf.FixedLenFeature([],tf.int64),
                                       'features':tf.FixedLenFeature([9],tf.float32)
                                   })

input_data = features['features']
#input_data = tf.reshape(input_data,[2,5])
lable_data = features['lable']

print features
print input_data
print lable_data

batch_size = 2
thread = 1
min_after_dequeue = 200
capacity = min_after_dequeue + (thread + 1) * batch_size


input_batch, label_batch = tf.train.maybe_shuffle_batch([input_data,lable_data],keep_input=True,
                                                  batch_size=batch_size,capacity=capacity,
                                                  num_threads=thread,min_after_dequeue=min_after_dequeue)

print 'here'


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    try:
        while not coord.should_stop():
            for i in range(20):
                val, l = sess.run([input_batch, label_batch])
                print val
                print l
            coord.request_stop()
    except tf.errors.OutOfRangeError:
        print ('Done training')
    finally:
        coord.request_stop()

coord.join(threads)
sess.close()


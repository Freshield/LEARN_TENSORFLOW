import tensorflow as tf
import numpy as np

writer = tf.python_io.TFRecordWriter('data/test.tfrecord')

for i in range(0,2):
    a = 0.618 + i
    b = [2016 + i, 2017 + i]
    c = np.array([[0,1,2],[3,4,5]]) + i
    c = c.astype(np.uint8)
    c_raw = c.tostring()
    print 'i:',i
    print 'a:',a
    print 'b:',b
    print 'c:',c
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {'a':tf.train.Feature(float_list=tf.train.FloatList(value=[a])),
                       'b':tf.train.Feature(int64_list=tf.train.Int64List(value=b)),
                       'c':tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_raw]))}
        )
    )
    serialized = example.SerializeToString()
    writer.write(serialized)
    print 'writer',i,'DOWN!'
writer.close()
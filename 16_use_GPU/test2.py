import tensorflow as tf

with tf.device('/gpu:1'):

    a = tf.constant([1.,2.,3.,4.,5.,6.], shape=[2,3], name='a')
    b = tf.constant([1.,2.,3.,4.,5.,6.], shape=[3,2], name='b')
    c = tf.matmul(a,b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print sess.run(c)
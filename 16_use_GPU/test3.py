import tensorflow as tf

c = []

for d in ['/gpu:0', '/gpu:1']:
    with tf.device(d):
        a = tf.constant([1.,2.,3.,4.,5.,6.],shape=[2,3])
        b = tf.constant([1.,2.,3.,4.,5.,6.],shape=[3,2])
        c.append(tf.matmul(a,b))

with tf.device('/cpu:0'):
    sum = tf.add_n(c)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print sess.run(sum)
import tensorflow as tf

v1 = tf.Variable(tf.zeros([10]), name='v1')
v2 = tf.Variable(tf.zeros([10]), name='v2')

init = tf.variables_initializer([v1])

saver = tf.train.Saver({'my_v2':v2})

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './model.ckpt')
    print ('Model restored')
    print sess.run(v1)
    print sess.run(v2)
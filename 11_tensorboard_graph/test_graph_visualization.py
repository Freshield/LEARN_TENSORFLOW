import tensorflow as tf

log_dir = 'logs/simple_test'

if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)
with tf.device('/gpu:0'):
    with tf.name_scope('hidden') as scope:
        a = tf.placeholder(tf.float32, [1, 2], name='a')
        tf.summary.scalar('a', a)
        W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='weights')
        tf.summary.scalar('weightss', W)
        b = tf.Variable(tf.ones([1, 1]), name='bias')
        tf.summary.scalar('biass', b)

with tf.device('/cpu:0'):
    result = tf.add(tf.matmul(a, W), b, name='result')
    test = tf.summary.scalar('result', result)

sess = tf.InteractiveSession()

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter(log_dir, sess.graph)

sess.run(tf.global_variables_initializer())

for i in xrange(500):
    if i % 10 == 0:
        res = sess.run(result, feed_dict={a:[[5,5]]})
        #summary = sess.run(merged, feed_dict={a:[[5,5]]})

        print ('result at step%s: %s' % (i, res))

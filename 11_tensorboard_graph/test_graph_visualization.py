import tensorflow as tf

log_dir = 'logs/simple_test'

if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)

with tf.name_scope('hidden') as scope:
    a = tf.constant(5, tf.float32, [1,2], 'alpha')
    tf.summary.scalar('alpha', tf.reduce_mean(a))
    W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='weights')
    tf.summary.scalar('weights', tf.reduce_mean(W))
    b = tf.Variable(tf.ones([1, 1]), name='bias')
    tf.summary.scalar('biass', tf.reduce_mean(b))
    result = tf.matmul(a, W) + b
    test = tf.summary.scalar('result', tf.reduce_mean(result))


sess = tf.InteractiveSession()

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter(log_dir, sess.graph)

sess.run(tf.global_variables_initializer())

for i in xrange(500):
    if i % 10 == 0:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, res = sess.run([merged, result],
                                options=run_options,
                                run_metadata=run_metadata)
        writer.add_run_metadata(run_metadata, 'step%d' % i)
        writer.add_summary(summary, i)

        print ('result at step%s: %s' % (i, res))

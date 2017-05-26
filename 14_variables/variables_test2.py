import tensorflow as tf

weights = tf.Variable(tf.ones([20]), name='weights')

w2 = tf.Variable(weights.initialized_value(), name='w2')

w_twice = tf.Variable(weights.initialized_value() * 2.0, name='w_twice')

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

print weights.eval()

print w2.eval()

print w_twice.eval()

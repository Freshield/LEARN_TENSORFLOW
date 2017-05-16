import tensorflow as tf


#for create convolution kernel
def conv2d(x, W, stride, padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

x = tf.ones([10,2,2,3])

filter = tf.Variable(tf.ones([1,1,3,5]))

y = conv2d(x, filter, 2, 'VALID')

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

a = [1,2,3]
b = [4,5,6]
c = []
c[0:0] = b
c[0:0] = a
print c

d = tf.ones([5, 8, 25, 1])
e = tf.pad(d, [[0,0],[2,2],[2,2],[0,0]], 'CONSTANT')
f = tf.nn.avg_pool(e, [1,3,3,1], [1,3,3,1], 'VALID')
print e.shape
print e.eval()[0,:,:,:]
print f.eval()[0,:,:,:]
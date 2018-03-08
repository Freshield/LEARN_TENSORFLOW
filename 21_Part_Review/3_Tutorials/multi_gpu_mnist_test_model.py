import tensorflow as tf

def _variable_on_cpu(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def get_weight_varible(name,shape):
    #return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return _variable_on_cpu(name, shape)

def get_bias_varible(name,shape):
    #return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return _variable_on_cpu(name, shape)
#filter_shape: [f_h, f_w, f_ic, f_oc]
def conv2d(layer_name, x, filter_shape):
    with tf.variable_scope(layer_name):
        w = get_weight_varible('w', filter_shape)
        b = get_bias_varible('b', filter_shape[-1])
        y = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME'), b)
        return y

def pool2d(layer_name, x):
    with tf.variable_scope(layer_name):
        y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return y

#inp_shape: [N, L]
#out_shape: [N, L]
def fc(layer_name, x, inp_shape, out_shape):
    with tf.variable_scope(layer_name):
        inp_dim = inp_shape[-1]
        out_dim = out_shape[-1]
        y = tf.reshape(x, shape=inp_shape)
        w = get_weight_varible('w', [inp_dim, out_dim])
        b = get_bias_varible('b', [out_dim])
        y = tf.add(tf.matmul(y, w), b)
        return y

def build_model(x, y):
    pred = tf.reshape(x,shape=[-1, 28, 28, 1])
    #layer 1
    pred = conv2d('conv_1', pred, [3, 3, 1, 8])
    pred = pool2d('pool_1', pred)
    #layer 2
    pred = conv2d('conv_2', pred, [3, 3, 8, 16])
    pred = pool2d('pool_2', pred)
    #layer fc
    pred = fc('fc', pred, [-1, 7*7*16], [-1, 10])

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    return pred, loss

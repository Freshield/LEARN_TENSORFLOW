import tensorflow as tf

#需要用get_variable来共享variable并把权重放到cpu
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
        para = []

        w = get_weight_varible('w', filter_shape)
        b = get_bias_varible('b', filter_shape[-1])
        y = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME'), b)

        para.append(w)
        para.append(b)

        return y, para

def pool2d(layer_name, x):
    with tf.variable_scope(layer_name):
        y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return y

#inp_shape: [N, L]
#out_shape: [N, L]
def fc(layer_name, x, inp_shape, out_shape):
    with tf.variable_scope(layer_name):
        para = []

        inp_dim = inp_shape[-1]
        out_dim = out_shape[-1]
        y = tf.reshape(x, shape=inp_shape)
        w = get_weight_varible('w', [inp_dim, out_dim])
        b = get_bias_varible('b', [out_dim])
        y = tf.add(tf.matmul(y, w), b)

        para.append(w)
        para.append(b)

        return y, para

def build_model(x, y, reg):
    para = []

    input_layer = tf.reshape(x,shape=[-1, 28, 28, 1])
    #layer 1
    conv1_layer, conv1_para = conv2d('conv_1', input_layer, [3, 3, 1, 8])
    para += conv1_para

    pool1_layer = pool2d('pool_1', conv1_layer)

    #layer 2
    conv2_layer, conv2_para = conv2d('conv_2', pool1_layer, [3, 3, 8, 16])
    para += conv2_para

    pool2_layer = pool2d('pool_2', conv2_layer)
    #layer fc
    pred, fc_para = fc('fc', pool2_layer, [-1, 7*7*16], [-1, 10])
    para += fc_para

    reg_loss = tf.zeros([])
    for p in para:
        reg_loss += reg * 0.5 * tf.nn.l2_loss(p)
    loss = reg_loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    return loss, accuracy

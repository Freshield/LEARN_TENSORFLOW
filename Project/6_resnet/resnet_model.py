import tensorflow as tf
import numpy as np


#create weights
def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  weight = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

  return weight


#create biases
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

#for create convolution kernel
def conv2d(x, W, stride, padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def batch_norm_layer(x, train_phase, scope_bn):
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = np.arange(len(x.shape) - 1)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def bn_relu_conv_same_layer(input_layer, filter_size, filter_depth, train_phase, name):
    with tf.variable_scope(name):
        with tf.variable_scope('brc_s'):
            input_depth = input_layer.shape[-1]
            bn_layer = batch_norm_layer(input_layer, train_phase, 'bn')
            relu_layer = tf.nn.relu(bn_layer, 'relu')
            filter = weight_variable([filter_size, filter_size, input_depth, filter_depth], 'filter')
            biases = bias_variable([filter_depth])
            conv_layer = conv2d(relu_layer, filter, 1, 'SAME') + biases
    return conv_layer, filter

def bn_relu_conv_half_layer(input_layer, filter_size, filter_depth, train_phase, name):
    #filter size check
    if filter_size != 1 and filter_size != 2:
        raise ValueError('filter size should be 1 or 2')
    with tf.variable_scope(name):
        with tf.variable_scope('brc_h'):
            input_depth = input_layer.shape[-1]
            bn_layer = batch_norm_layer(input_layer, train_phase, 'bn')
            relu_layer = tf.nn.relu(bn_layer, 'relu')
            filter = weight_variable([filter_size, filter_size, input_depth, filter_depth], 'filter')
            biases = bias_variable([filter_depth])
            conv_layer = conv2d(relu_layer, filter, 2, 'VALID') + biases
    return conv_layer, filter

def resnet_same_block(input_layer, train_phase):
    input_depth = input_layer.shape[-1]
    #check
    if input_depth < 4:
        raise TypeError('input depth is too small')
    small_depth = input_depth // 4
    with tf.variable_scope('res_sb'):
        block_layer1, f1 = bn_relu_conv_same_layer(input_layer, 1, small_depth, train_phase, 'bl_1')
        block_layer2, f2 = bn_relu_conv_same_layer(block_layer1, 3, small_depth, train_phase, 'bl_2')
        block_layer3, f3 = bn_relu_conv_same_layer(block_layer2, 1, input_depth, train_phase, 'bl_3')
        add_layer = input_layer + block_layer3
        parameters = (f1, f2, f3)
    return add_layer, parameters


def resnet_diffD_sameS_block(input_layer, block_depth, train_phase):
    input_depth = input_layer.shape[-1]
    #check
    if input_depth < 4:
        raise TypeError('input depth is too small')
    small_depth = block_depth // 4
    with tf.variable_scope('res_dsb'):
        block_layer1, f1 = bn_relu_conv_same_layer(input_layer, 1, small_depth, train_phase, 'bl_1')
        block_layer2, f2 = bn_relu_conv_same_layer(block_layer1, 3, small_depth, train_phase, 'bl_2')
        block_layer3, f3 = bn_relu_conv_same_layer(block_layer2, 1, block_depth, train_phase, 'bl_3')
        block_layer4, f4 = bn_relu_conv_same_layer(input_layer, 1, block_depth, train_phase, 'bl_4')
        add_layer = block_layer4 + block_layer3
        parameters = (f1, f2, f3, f4)
    return add_layer, parameters


def resnet_diffD_halfS_block(input_layer, block_depth, train_phase):
    input_depth = input_layer.shape[-1]
    #check
    if input_depth < 4:
        raise TypeError('input depth is too small')
    small_depth = block_depth // 4
    with tf.variable_scope('res_dhb'):
        block_layer1, f1 = bn_relu_conv_half_layer(input_layer, 1, small_depth, train_phase, 'bl_1')
        block_layer2, f2 = bn_relu_conv_same_layer(block_layer1, 3, small_depth, train_phase, 'bl_2')
        block_layer3, f3 = bn_relu_conv_same_layer(block_layer2, 1, block_depth, train_phase, 'bl_3')
        block_layer4, f4 = bn_relu_conv_half_layer(input_layer, 1, block_depth, train_phase, 'bl_4')
        add_layer = block_layer4 + block_layer3
        parameters = (f1, f2, f3, f4)
    return add_layer, parameters

def resnet_first_layer(input_layer, layer_depth, train_phase, name):
    parameters = []
    with tf.variable_scope(name):
        with tf.variable_scope('res_1st'):
            layer1, p1 = resnet_diffD_sameS_block(input_layer, layer_depth, train_phase)
            layer2, p2 = resnet_same_block(layer1, train_phase)
    parameters[0:0] = p1
    parameters[0:0] = p2
    return layer2, parameters

def resnet_layer(input_layer, layer_depth, train_phase, name):
    parameters = []
    with tf.variable_scope(name):
        with tf.variable_scope('res'):
            layer1, p1 = resnet_diffD_halfS_block(input_layer, layer_depth, train_phase)
            layer2, p2 = resnet_same_block(layer1, train_phase)
    parameters[0:0] = p1
    parameters[0:0] = p2
    return layer2, parameters

def fc_layer(input_layer, label_size):
    with tf.variable_scope('fc'):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, label_size], 'fc_weight')
        b = bias_variable([label_size])
        output = tf.matmul(input_layer, W) + b
    return output, [W]

def inference(input_layer, train_phase):
    parameters = []
    #input shape should be (N,32,100,3)
    input_depth = input_layer.shape[-1]

    #input[N,32,100,3],output[N,32,100,64]
    with tf.variable_scope('preprocess'):
        bn_input = batch_norm_layer(input_layer, train_phase, 'bn_input')
        filter = weight_variable([3,3,input_depth,64], 'filter_input')
        biases = bias_variable([64])
        conv_input = conv2d(bn_input, filter, 1, 'SAME') + biases

    #input[N,32,100,64],output[N,32,100,256]
    resnet_l1, p1 = resnet_first_layer(conv_input, 256, train_phase, 'resnet_l1')
    parameters[0:0] = p1

    #input[N,32,100,256],output[N,16,50,512]
    resnet_l2, p2 = resnet_layer(resnet_l1, 512, train_phase, 'resnet_l2')
    parameters[0:0] = p2

    #input[N,16,50,512],output[N,8,25,1024]
    resnet_l3, p3 = resnet_layer(resnet_l2, 1024, train_phase, 'resnet_l3')
    parameters[0:0] = p3

    #pad for avg pool
    #input[N,8,25,1024],output[N,10,27,1024]
    resnet_l3_pad = tf.pad(resnet_l3, [[0,0],[2,2],[2,2],[0,0]], 'CONSTANT', 'l3_pad')
    #input[N,10,27,1024],output[N,4,9,1024]
    avg_pool_layer = tf.nn.avg_pool(resnet_l3_pad, [1,3,3,1], [1,3,3,1], 'VALID')

    #platten for fc
    #input[N,10,27,1024],output[N,4*9*1024]
    avg_pool_flat = tf.reshape(avg_pool_layer, [-1, 4 * 9 * 1024])

    #fc layer
    #input[N,4*9*1024],output[N,3]
    y_pred, p_fc = fc_layer(avg_pool_flat, 3)
    parameters[0:0] = p_fc

    return y_pred, parameters




x = tf.ones([100,32,100,3],dtype=tf.float32)

train_phase = tf.placeholder(tf.bool)

y, para = inference(x, train_phase)

print y.shape
print len(para)

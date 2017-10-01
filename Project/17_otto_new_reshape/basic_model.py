import time
import numpy as np
import signal

import tensorflow as tf


#create weights
#ver 1.0
def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  weight = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

  return weight


#create biases
#ver 1.0
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

#for create convolution kernel
#ver 1.0
def conv2d(x, W, stride, padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


# for create the pooling
#ver 1.0
def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# for create batch norm layer
#ver 1.0
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

#dense layer
#ver 1.0
def dense_layer(input_layer, output_size, name, act=tf.nn.relu):
    with tf.variable_scope(name):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, output_size], 'dense_weight')
        b = bias_variable([output_size])
        output = act(tf.matmul(input_layer, W) + b)
    return output, [W]


# conv-bn-relu-maxpooling
#ver 1.0
def conv_bn_pool_layer(input_layer, filter_depth, train_phase, name, filter_size=3):
    input_depth = input_layer.shape[-1]
    with tf.variable_scope(name):
        filter = weight_variable([filter_size, filter_size, input_depth, filter_depth], "filter")
        biases = bias_variable([filter_depth])
        conv_output = conv2d(input_layer, filter, 1, "SAME") + biases
        bn_output = batch_norm_layer(conv_output, train_phase, "conv_bn")
        act_output = tf.nn.relu(bn_output)
        output = max_pool_2x2(act_output)
    return output, [filter]


#the fully connect layer
#ver 1.0
def fc_layer(input_layer, label_size):
    with tf.variable_scope('fc'):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, label_size], 'fc_weight')
        b = bias_variable([label_size])
        output = tf.matmul(input_layer, W) + b
    return output, [W]

# fully connected layer
# fc-bn-relu-drop
#ver 1.0
def fc_bn_drop_layer(input_layer, output_size, train_phase, keep_prob, name):
    with tf.variable_scope(name):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, output_size], 'fc_weight')
        b = bias_variable([output_size])
        fc_out = tf.matmul(input_layer, W) + b
        bn_out = batch_norm_layer(fc_out, train_phase, "fc_bn")
        act_out = tf.nn.relu(bn_out)
        output = tf.nn.dropout(act_out, keep_prob)
    return output, [W]

# score layer
#ver 1.0
def score_layer(input_layer, label_size):
    with tf.variable_scope("score"):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, label_size], 'score_weight')
        b = bias_variable([label_size])
        output = tf.matmul(input_layer, W) + b
    return output, [W]

#get the correct number and accuracy
#ver 1.0
def corr_num_acc(labels, logits):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_num, accuracy

#get the loss
#ver 1.0
def loss(labels, logits, reg=None, parameters=None):
    with tf.variable_scope("loss"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy'))

        if parameters == None:
            cost = cross_entropy
        else:
            reg_loss = 0.0
            for para in parameters:
                reg_loss += reg * 0.5 * tf.nn.l2_loss(para)
            cost = cross_entropy + reg_loss
    return cost


#to get the random hypers for cross valid
#ver 1.0
def random_uniform_array(number, start, end):
    array = np.zeros(number)
    for i in np.arange(number):
        array[i] = 10 ** np.random.uniform(start, end)

    return array

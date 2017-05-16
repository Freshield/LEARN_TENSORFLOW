import tensorflow as tf
import numpy as np
import os
from keras.utils import np_utils

############################################################
############# helpers ######################################
############################################################

#split the dataset into three part:
#training, validation, test
def split_dataset(dataset, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(dataset))

    train_set = dataset.values[0:-test_dataset_size * 2]
    validation_set = dataset.values[-test_dataset_size * 2:-test_dataset_size]
    test_set = dataset.values[-test_dataset_size:len(dataset)]

    return train_set, validation_set, test_set

def normalize_dataset(dataset, mean_value=None):
    norm_dataset = np.zeros((dataset.shape))
    norm_dataset[:, :] = dataset[:, :]

    if mean_value == None:
        mean_value = np.mean(norm_dataset[:,:200])

    norm_dataset[:,:200] -= mean_value

    return norm_dataset, mean_value

def reshape_dataset(dataset, SPAN):
    input_data = dataset[:, :241]

    output_data = dataset[:, 240 + SPAN[0]]
    output_data = np_utils.to_categorical(output_data)

    return input_data, output_data

def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs


def sequence_get_data(X_dataset, y_dataset, indexs, last_index, batch_size):
    next_index = last_index + batch_size
    out_of_dataset = False

    if next_index > X_dataset.shape[0]:
        next_index -= X_dataset.shape[0]
        last_part = np.arange(last_index, indexs.shape[0])
        before_part = np.arange(next_index)
        span_index = indexs[np.concatenate((last_part, before_part))]
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]

    X_data = X_dataset[span_index]
    y_data = y_dataset[span_index]
    return (next_index, {'X':X_data, 'y':y_data}, out_of_dataset)




###########################################################
################# graph helper ############################
###########################################################

#create weights
def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  weight = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
  #tf.random_normal_initializer()
  #tf.contrib.layers.xavier_initializer()
  return weight


#create biases
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def hidden_layer(input_layer, hidden_size, name, act=tf.nn.sigmoid):
    with tf.variable_scope(name):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, hidden_size], 'weights')
        b = bias_variable([hidden_size])
        output = act(tf.matmul(input_layer, W) + b)
    return output, [W]

def fc_layer(input_layer, label_size, name):
    with tf.variable_scope(name):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, label_size], 'weights')
        b = bias_variable([label_size])
        output = tf.matmul(input_layer, W) + b
    return output, [W]


def inference(input_layer, act=tf.nn.sigmoid):
    parameters = []
    #input shape should be (N,241)

    h1_layer, p1 = hidden_layer(input_layer, 50, 'h1', act)
    parameters[0:0] = p1

    h2_layer, p2 = hidden_layer(h1_layer, 40, 'h2', act)
    parameters[0:0] = p2

    h3_layer, p3 = hidden_layer(h2_layer, 30, 'h3', act)
    parameters[0:0] = p3

    y_pred, p4 = fc_layer(h3_layer, 3, 'scores')
    parameters[0:0] = p4

    return y_pred, parameters

def loss(labels, logits, reg=None, parameters=None):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy'))

    if parameters == None:
        cost = cross_entropy
    else:
        reg_loss = 0.0
        for para in parameters:
            reg_loss += reg * 0.5 * tf.nn.l2_loss(para)
        cost = cross_entropy + reg_loss

    return cost

def corr_num_acc(labels, logits):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_num, accuracy

def do_eval(sess, X_dataset, y_dataset, batch_size, correct_num, placeholders, merged=None, test_writer=None, global_step=None):
    input_x, input_y = placeholders
    num_epoch = X_dataset.shape[0] // batch_size
    rest_data_size = X_dataset.shape[0] % batch_size

    index = 0
    count = 0.0
    indexs = np.arange(X_dataset.shape[0])

    for step in xrange(num_epoch):
        index, data, _ = sequence_get_data(X_dataset, y_dataset, indexs, index, batch_size)

        feed_dict = {input_x:data['X'], input_y:data['y']}
        if step != num_epoch -1 or merged == None:
            num = sess.run(correct_num, feed_dict=feed_dict)
        else:
            summary, num = sess.run([merged, correct_num], feed_dict=feed_dict)
            #test_writer.add_summary(summary, global_step)
        count += num

    if rest_data_size != 0:
        index, data, _ = sequence_get_data(X_dataset, y_dataset, indexs, index, rest_data_size)
        feed_dict = {input_x: data['X'], input_y: data['y']}
        num = sess.run(correct_num, feed_dict=feed_dict)
        count += num
    return count / X_dataset.shape[0]



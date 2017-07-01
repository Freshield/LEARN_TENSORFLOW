import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os
############################################################
############# helpers ######################################
############################################################
def get_batch_data(data_set, batch_size):
    lines_num = data_set.shape[0] - 1
    random_index = np.random.randint(lines_num, size=[batch_size])
    columns = data_set.values[random_index]
    features = columns[:, :-1]
    labels = columns[:, -1]
    return {'features': features, 'labels': labels}

def get_whole_data(data_set):
    features = data_set.values[:, :-1]
    labels = data_set.values[:, -1]
    return {'features': features, 'labels': labels}

def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    #index = tf.random_shuffle(tf.range(0, data_size))
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs

def sequence_get_data(data_set, indexs, last_index, batch_size):
    next_index = last_index + batch_size
    out_of_dataset = False

    if next_index > data_set.shape[0]:
        last_index -= data_set.shape[0]
        next_index -= data_set.shape[0]
        out_of_dataset = True

    span_index = indexs[last_index:next_index]

    columns = data_set.values[span_index]
    features = columns[:, :-1]
    labels = columns[:, -1]
    return (next_index, {'features': features, 'labels': labels}, out_of_dataset)

def conv2d(x, W, stride, padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#input
x = tf.placeholder(tf.float32, [None, 4])
xr = tf.reshape(x, shape=[-1, 1, 4, 1])
x_rows = tf.concat([xr, xr], axis=2)
x_ = tf.concat([x_rows for i in range(8)], axis=1)
y_ = tf.placeholder(tf.int32, [None])
y_one_hot = tf.one_hot(y_, 3)

print x_

#neural

W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_, W_conv1, 1, 'SAME') + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1, 'SAME') + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)

print h_pool2

W_fc1 = weight_variable([2 * 2 * 64, 256])
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([256, 3])
b_fc2 = bias_variable([3])
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=y))

train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_one_hot, 1))
correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())


train_dataset = pd.read_csv('iris_training.csv', header=None)
test_dataset = pd.read_csv('iris_test.csv', header=None)

indexs = get_random_seq_indexs(train_dataset)
out_of_dataset = False
last_index = 0
batch_size = 30

data = get_whole_data(train_dataset)

for step in xrange(15000):

    if out_of_dataset == True:
        indexs = get_random_seq_indexs(train_dataset)
        last_index = 0
        out_of_dataset = False

    last_index, data, out_of_dataset = sequence_get_data(train_dataset, indexs, last_index, batch_size)
    #data = get_batch_data(train_dataset, batch_size)

    _, loss_v = sess.run([train_step, cross_entropy],
                         feed_dict={x: data['features'], y_: data['labels'], keep_prob: 0.5})

    if step % 100 == 0:
        print 'loss in step %d is %f' % (step, loss_v)

    if step % 500 == 0 or step == 5000 - 1:

        data = get_whole_data(train_dataset)

        result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob: 1.0})
        print '----------train acc in step %d is %f-------------' % (step, result)

        data = get_whole_data(test_dataset)

        result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob: 1.0})
        print '----------test acc in step %d is %f-------------' % (step, result)



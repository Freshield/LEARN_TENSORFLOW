import pandas as pd
import tensorflow as tf
import numpy as np
import time

filename = '/home/freshield/ciena_test/FiberID_Data.csv'

batch = 100
lr_rate = 0.01
max_step = 10000
reg = 0.01


def split_dataset(dataset, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(dataset))

    train_set = dataset[0:-test_dataset_size]
    test_set = dataset[-test_dataset_size:len(dataset)]


    return train_set, test_set


def get_batch_data(data_set, batch_size):
    lines_num = data_set.shape[0] - 1
    random_index = np.random.randint(lines_num, size=[batch_size])
    features = data_set.values[random_index, :-20]
    labels = data_set.values[random_index, -1]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return {'features': features, 'labels': labels_one_hot}


def get_whole_data(data_set):
    features = data_set.values[:, :-20]
    labels = data_set.values[:, -1]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return {'features': features, 'labels': labels_one_hot}

def sequence_get_data(data_set, last_index, batch_size):
    next_index = last_index + batch_size
    if next_index > len(data_set):
        last_index -= len(data_set)
        next_index -= len(data_set)
    indexs = np.arange(last_index, next_index, 1)

    features = data_set.values[indexs, :-20]
    labels = data_set.values[indexs, -1]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return (next_index, {'features': features, 'labels': labels_one_hot})

def do_eval(sess, data_set, batch_size):
    num_epoch = len(data_set) / batch_size
    reset_data_size = len(data_set) % batch_size

    index = 0
    count = 0.0
    for step in xrange(num_epoch):
        index, data = sequence_get_data(data_set, index, batch_size)
        result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob:1.0})
        print ('accuracy in step %d is %f' % (step, result))
        count += result * batch_size
    if reset_data_size != 0:
        #the reset data
        index, data = sequence_get_data(data_set, index, reset_data_size)
        result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob:1.0})
        count += result * reset_data_size
    return count / len(data_set)
#####################################################################
############### create the graph ####################################
#####################################################################

dataset = pd.read_csv(filename, header=None)
train_dataset, test_dataset = split_dataset(dataset, radio=0.1)

print train_dataset.shape
print test_dataset.shape

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 241], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, 3], name='input_y')

with tf.name_scope('hidden1'):
    W1 = tf.Variable(tf.truncated_normal([241, 100], stddev=0.35), name='weights')
    b1 = tf.Variable(tf.zeros([100]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

with tf.name_scope('hidden2'):
    W2 = tf.Variable(tf.truncated_normal([100, 50], stddev=0.35), name='weights')
    b2 = tf.Variable(tf.zeros([50]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

with tf.name_scope('hidden3'):
    W3 = tf.Variable(tf.truncated_normal([50, 30], stddev=0.35), name='weights')
    b3 = tf.Variable(tf.zeros([30]), name='biases')
    hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    hidden3_drop = tf.nn.dropout(hidden3, keep_prob)

with tf.name_scope('scores'):
    W4 = tf.Variable(tf.truncated_normal([30, 3], stddev=0.35), name='weights')
    b4 = tf.Variable(tf.zeros([3]), name='biases')
    y = tf.matmul(hidden3_drop, W4) + b4

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name='xentropy')
    loss = (cross_entropy + reg * tf.nn.l2_loss(W1) + reg * tf.nn.l2_loss(b1) +
            tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) +
            tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3) +
            tf.nn.l2_loss(W4) + tf.nn.l2_loss(b4))

train_step = tf.train.AdamOptimizer(lr_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver()

#################################################################
################## train part ###################################
#################################################################

path = 'modules/0.99/model.ckpt'
saver.restore(sess, path)
print "Model restored."

result = do_eval(sess, test_dataset, batch)
print 'last accuracy is %f' % (result)

sess.close()
""""""
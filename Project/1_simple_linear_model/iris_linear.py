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

train_dataset = pd.read_csv('iris_training.csv', header=None)
test_dataset = pd.read_csv('iris_test.csv', header=None)

#input
x = tf.placeholder(tf.float32, [None, 4])
y_ = tf.placeholder(tf.int32, [None])
y_one_hot = tf.one_hot(y_, 3)

#neural

W1 = tf.Variable(tf.truncated_normal([4, 100], stddev=0.35))
b1 = tf.Variable(tf.zeros([100]))
h1 = tf.matmul(x, W1) + b1

W2 = tf.Variable(tf.truncated_normal([100, 200], stddev=1.0))
b2 = tf.Variable(tf.zeros([200]))
h2 = tf.matmul(h1, W2) + b2

keep_prob = tf.placeholder(tf.float32)
h2_drop = tf.nn.dropout(h2, keep_prob=keep_prob)

W3 = tf.Variable(tf.truncated_normal([200, 3], stddev=1.0))
b3 = tf.Variable(tf.zeros([3]))
y = tf.matmul(h2_drop, W3) + b3

cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=y))

train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in xrange(15000):
    data = get_batch_data(train_dataset, 30)

    _, loss_v = sess.run([train_step, cross_entropy],
                         feed_dict={x: data['features'], y_: data['labels'], keep_prob: 0.5})

    if step % 100 == 0:
        print 'loss in step %d is %f' % (step, loss_v)

    if step % 500 == 0 or step == 5000 - 1:
        result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob: 1.0})
        print '----------train acc in step %d is %f-------------' % (step, result)

        data = get_whole_data(test_dataset)

        result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob: 1.0})
        print '----------test acc in step %d is %f-------------' % (step, result)



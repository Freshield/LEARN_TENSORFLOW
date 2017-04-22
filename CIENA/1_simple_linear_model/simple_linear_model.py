import pandas as pd
import tensorflow as tf
import numpy as np
import time

filename = 'ciena.csv'

batch = 100
lr_rate = 0.01
max_step = 3000
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
    features = data_set.values[random_index, :-1]
    labels = data_set.values[random_index, -1]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return {'features': features, 'labels': labels_one_hot}


def get_whole_data(data_set):
    features = data_set.values[:, :-1]
    labels = data_set.values[:, -1]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return {'features': features, 'labels': labels_one_hot}

#####################################################################
############### create the graph ####################################
#####################################################################

dataset = pd.read_csv(filename, header=None)
train_dataset, test_dataset = split_dataset(dataset, radio=0.3)

print train_dataset.shape
print test_dataset.shape

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 6260], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, 3], name='input_y')

with tf.name_scope('hidden1'):
    W1 = tf.Variable(tf.truncated_normal([6260, 10000], stddev=1.0), name='weights')
    b1 = tf.Variable(tf.zeros([10000]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

with tf.name_scope('hidden2'):
    W2 = tf.Variable(tf.truncated_normal([10000, 3000], stddev=1.0), name='weights')
    b2 = tf.Variable(tf.zeros([3000]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

with tf.name_scope('scores'):
    W3 = tf.Variable(tf.truncated_normal([3000, 3], stddev=1.0), name='weights')
    b3 = tf.Variable(tf.zeros([3]), name='biases')
    y = tf.matmul(hidden2, W3) + b3

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name='xentropy')
    loss = (cross_entropy + reg * tf.nn.l2_loss(W1) + reg * tf.nn.l2_loss(b1) +
            tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3))

train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for step in range(max_step):
    before_time = time.time()
    data = get_batch_data(train_dataset, batch)
    # data = get_whole_data(train_dataset)

    _, loss_v = sess.run([train_step, loss],
                       feed_dict={x: data['features'], y_: data['labels']})

    if step % 100 == 0:
        print 'loss in step %d is %f' % (step, loss_v)

        last_time = time.time()
        span_time = last_time - before_time
        print ('rest time is %f minutes' % (span_time * (max_step - step) / 60))
    if step % 1000 == 0:
        data = get_batch_data(train_dataset, batch)
        result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels']})
        print 'accuracy in step %d is %f' % (step, result)
        # Test trained model

data = get_batch_data(train_dataset, 300)
result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels']})
print 'last accuracy is %f' % (result)
""""""
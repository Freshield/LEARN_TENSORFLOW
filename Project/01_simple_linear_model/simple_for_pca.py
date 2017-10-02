import pandas as pd
import tensorflow as tf
import numpy as np
import time

filename = 'ciena_test.csv'

batch = 100
lr_rate = 0.01
max_step = 10000
reg = 0.01
lr_decay = 0.99
lr_decay_epoch = 1500


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
    labels = data_set.values[random_index, -2]
    np_labels = np.array(labels, dtype=np.int32)
    labels_one_hot = np.eye(3)[np_labels]
    return {'features': features, 'labels': labels_one_hot}


def get_whole_data(data_set):
    features = data_set.values[:, :-20]
    labels = data_set.values[:, -2]
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
    labels = data_set.values[indexs, -2]
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
    W1 = tf.Variable(tf.truncated_normal([241, 200], stddev=1.0), name='weights')
    b1 = tf.Variable(tf.zeros([200]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

with tf.name_scope('hidden2'):
    W2 = tf.Variable(tf.truncated_normal([200, 150], stddev=1.0), name='weights')
    b2 = tf.Variable(tf.zeros([150]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

with tf.name_scope('hidden3'):
    W3 = tf.Variable(tf.truncated_normal([150, 30], stddev=1.0), name='weights')
    b3 = tf.Variable(tf.zeros([30]), name='biases')
    hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    hidden3_drop = tf.nn.dropout(hidden3, keep_prob=keep_prob)

with tf.name_scope('scores'):
    W4 = tf.Variable(tf.truncated_normal([30, 3], stddev=1.0), name='weights')
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

index = 0
last_accuracy = 0.6
# Train
for step in range(max_step):
    before_time = time.time()
    data = get_batch_data(train_dataset, batch)
    # data = get_whole_data(train_dataset)
    #index, data = sequence_get_data(train_dataset, index, batch)

    _, loss_v = sess.run([train_step, cross_entropy],
                       feed_dict={x: data['features'], y_: data['labels'], keep_prob:1.0})

    if step % 100 == 0:
        print 'loss in step %d is %f' % (step, loss_v)

        last_time = time.time()
        span_time = last_time - before_time
        print ('100 steps is %f second' % (span_time * 100))
        print ('rest time is %f minutes' % (span_time * (max_step - step) / 60))
    if step % 500 == 0:
        #data = get_batch_data(test_dataset, 1000)
        #result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob:1.0})
        result = do_eval(sess, test_dataset, batch)
        print '----------accuracy in step %d is %f-------------' % (step, result)
        if result > last_accuracy:
            last_accuracy = result
            path = "modules/%.2f/model.ckpt" % result
            if tf.gfile.Exists(path):
                tf.gfile.DeleteRecursively(path)
            tf.gfile.MakeDirs(path)
            save_path = saver.save(sess, path)
            print("Model saved in file: %s" % save_path)
        # Test trained model

    #if step > 0 and step % lr_decay_epoch == 0:
        #lr_rate *= lr_decay


result = do_eval(sess, test_dataset, batch)
print '-----------last accuracy is %f------------' % (result)

sess.close()
""""""
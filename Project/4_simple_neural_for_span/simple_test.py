import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

def split_dataset(dataset, radio):
    test_dataset_size = int(radio * len(dataset))
    data_size = dataset.shape[0]
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    train_set = dataset.values[indexs[:-test_dataset_size * 2]]
    validation_set = dataset.values[indexs[-test_dataset_size * 2 : -test_dataset_size]]
    test_set = dataset.values[indexs[-test_dataset_size : ]]
    return train_set, validation_set, test_set

def normalize_dataset(dataset):
    norm_dataset = np.zeros((dataset.shape))
    norm_dataset[:,:] = dataset[:,:]
    #norm_dataset[:,:241] = norm_dataset[:,:241] - np.mean(norm_dataset[:,:241])
    #norm_dataset[:,:241] = norm_dataset[:,:241] - np.amin(norm_dataset[:,:241])
    #norm_dataset[:,:241] = norm_dataset[:,:241] / np.amax(norm_dataset[:,:241])

    return norm_dataset

def get_batch_data(data_set, batch_size):
    random_index = np.random.randint(data_set.shape[0], size=[batch_size])
    columns = data_set[random_index]
    features = columns[:,:241]
    labels = columns[:,-2]
    return {'features':features, 'labels':labels}

def get_whole_data(data_set):
    features = data_set[:,:241]
    labels = data_set[:,-2]
    return {'features':features, 'labels':labels}
#-----------------------------
filename = 'ciena_test.csv'
dataset = pd.read_csv(filename, header=None)
train_dataset, validation_dataset, test_dataset = split_dataset(dataset, 0.1)
train_dataset = normalize_dataset(train_dataset)
validation_dataset = normalize_dataset(validation_dataset)
test_dataset = normalize_dataset(test_dataset)

reg = 1e-4
lr_rate = 0.01
max_step = 100000

def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  weight = tf.get_variable(name, shape,initializer=tf.contrib.layers.xavier_initializer(uniform=False))
  return weight

def batch_norm_layer1(x,train_phase,scope_bn):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=True,
    reuse=None,
    trainable=True,
    scope=scope_bn)

    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=False,
    reuse=True,
    trainable=True,
    scope=scope_bn)

    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z

with tf.Graph().as_default():
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [None, 241])
        y_ = tf.placeholder(tf.int32, [None])
        y_one_hot = tf.one_hot(y_, 3)
        train_phase = tf.placeholder(tf.bool, name='train_phase')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        bn_input = batch_norm_layer1(x, train_phase, 'bn_input')

        W1 = weight_variable([241, 1024], 'W1')
        b1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        h1 = tf.matmul(bn_input, W1) + b1
        bn_h1 = batch_norm_layer1(h1, train_phase, 'bn_h1')
        act_h1 = tf.nn.relu(bn_h1)
        drop_h1 = tf.nn.dropout(act_h1, keep_prob=keep_prob)

        W2 = weight_variable([1024, 512], 'W2')
        b2 = tf.Variable(tf.constant(0.1, shape=[512]))
        h2 = tf.matmul(drop_h1, W2) + b2
        bn_h2 = batch_norm_layer1(h2, train_phase, 'bn_h2')
        act_h2 = tf.nn.relu(bn_h2)
        drop_h2 = tf.nn.dropout(act_h2, keep_prob=keep_prob)

        W3 = weight_variable([512, 256], 'W3')
        b3 = tf.Variable(tf.constant(0.1, shape=[256]))
        h3 = tf.matmul(drop_h2, W3) + b3
        bn_h3 = batch_norm_layer1(h3, train_phase, 'bn_h3')
        act_h3 = tf.nn.relu(bn_h3)
        drop_h3 = tf.nn.dropout(act_h3, keep_prob=keep_prob)

        W4 = weight_variable([256, 128], 'W4')
        b4 = tf.Variable(tf.constant(0.1, shape=[128]))
        h4 = tf.matmul(drop_h3, W4) + b4
        bn_h4 = batch_norm_layer1(h4, train_phase, 'bn_h4')
        act_h4 = tf.nn.relu(bn_h4)
        drop_h4 = tf.nn.dropout(act_h4, keep_prob=keep_prob)

        W5 = weight_variable([128, 3], 'W5')
        b5 = tf.Variable(tf.constant(0.1, shape=[3]))
        y = tf.matmul(drop_h4, W5) + b5

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=y))

        reg_loss = 0.5 * reg * (
            tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(
                W3) + tf.nn.l2_loss(b3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(b4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(b5))

        loss = cross_entropy + reg_loss

        train_op = tf.train.AdamOptimizer(lr_rate).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_one_hot, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())

        for step in xrange(max_step):
            data = get_batch_data(train_dataset, 100)

            feed_dict = {x:data['features'], y_:data['labels'], train_phase:True, keep_prob:0.5}

            _, loss_v, acc = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)

            if step % 100 == 0:
                print '---loss in step %d is %f, acc is %.3f---' % (step, loss_v, acc)

            if step % 500 == 0:
                data = get_whole_data(train_dataset)

                feed_dict = {x: data['features'], y_: data['labels'], train_phase: False, keep_prob: 1.0}

                result = sess.run(accuracy, feed_dict=feed_dict)

                print '--------train acc in step %d is %f--------' % (step, acc)

                data = get_whole_data(validation_dataset)

                feed_dict = {x: data['features'], y_: data['labels'], train_phase:False, keep_prob:1.0}

                result = sess.run(accuracy, feed_dict=feed_dict)

                print '--------valida acc in step %d is %f--------' % (step, result)

            if step == max_step - 1:
                data = get_whole_data(test_dataset)

                feed_dict = {x: data['features'], y_: data['labels'], train_phase:False, keep_prob:1.0}

                result = sess.run(accuracy, feed_dict=feed_dict)

                print '--------test acc in step %d is %f--------' % (step, result)

            if step % 1000 == 0:
                lr_rate = lr_rate * 0.99



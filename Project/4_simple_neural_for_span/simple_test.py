import tensorflow as tf
import pandas as pd
import numpy as np

def split_dataset(dataset, radio):
    test_dataset_size = int(radio * len(dataset))
    data_size = dataset.shape[0]
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    train_set = dataset.values[indexs[:-test_dataset_size * 2]]
    validation_set = dataset.values[indexs[-test_dataset_size * 2 : -test_dataset_size]]
    test_set = dataset.values[indexs[-test_dataset_size : ]]
    return train_set, validation_set, test_set

def get_batch_data(data_set, batch_size):
    random_index = np.random.randint(data_set.shape[0], size=[batch_size])
    columns = data_set[random_index]
    features = columns[:,:241]
    labels = columns[:,259]
    return {'features':features, 'labels':labels}

def get_whole_data(data_set):
    features = data_set[:,:241]
    labels = data_set[:,259]
    return {'features':features, 'labels':labels}
#-----------------------------
filename = 'ciena_test.csv'
dataset = pd.read_csv(filename, header=None)
train_dataset, validation_dataset, test_dataset = split_dataset(dataset, 0.1)

reg = 0.01
lr_rate = 0.002
max_step = 100000

with tf.Graph().as_default():
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [None, 241])
        y_ = tf.placeholder(tf.int32, [None])
        y_one_hot = tf.one_hot(y_, 3)

        W1 = tf.Variable(tf.truncated_normal([241, 1024], stddev=1.0))
        b1 = tf.Variable(tf.constant(1.0, shape=[1024]))
        h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

        W2 = tf.Variable(tf.truncated_normal([1024, 512], stddev=1.0))
        b2 = tf.Variable(tf.constant(1.0, shape=[512]))
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

        W3 = tf.Variable(tf.truncated_normal([512, 256], stddev=1.0))
        b3 = tf.Variable(tf.constant(1.0, shape=[256]))
        h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

        W4 = tf.Variable(tf.truncated_normal([256, 128], stddev=1.0))
        b4 = tf.Variable(tf.constant(1.0, shape=[128]))
        h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)

        W5 = tf.Variable(tf.truncated_normal([128, 3], stddev=1.0))
        b5 = tf.Variable(tf.constant(1.0, shape=[3]))
        y = tf.matmul(h4, W5) + b5

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

            feed_dict = {x:data['features'], y_:data['labels']}

            _, loss_v, acc = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)

            if step % 100 == 0:
                print '---loss in step %d is %f---' % (step, loss_v)

            if step % 500 == 0:
                print '--------train acc in step %d is %f--------' % (step, acc)
                data = get_whole_data(validation_dataset)

                feed_dict = {x: data['features'], y_: data['labels']}

                result = sess.run(accuracy, feed_dict=feed_dict)

                print '--------valida acc in step %d is %f--------' % (step, result)

            if step == max_step - 1:
                data = get_whole_data(test_dataset)

                feed_dict = {x: data['features'], y_: data['labels']}

                result = sess.run(accuracy, feed_dict=feed_dict)

                print '--------test acc in step %d is %f--------' % (step, result)




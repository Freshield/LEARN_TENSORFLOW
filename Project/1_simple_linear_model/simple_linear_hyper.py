import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os
############################################################
############# helpers ######################################
############################################################
def copyFiles(sourceDir,  targetDir):
    if sourceDir.find(".csv") > 0:
        print 'error'
        return
    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,  file)
        targetFile = os.path.join(targetDir,  file)
        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or(os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):
                    open(targetFile, "wb").write(open(sourceFile, "rb").read())
        if os.path.isdir(sourceFile):
            First_Directory = False
            copyFiles(sourceFile, targetFile)

def split_dataset(dataset, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(dataset))

    train_set = dataset[0:-test_dataset_size * 2]
    validation_set = dataset[-test_dataset_size * 2:-test_dataset_size]
    test_set = dataset[-test_dataset_size:len(dataset)]


    return train_set, validation_set, test_set


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

#####################################################################
############### create the graph ####################################
#####################################################################
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def inference(hidden1_size, hidden2_size, hidden3_size, lr_rate, reg, stddev, use_L2=True):

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 241], name='input_x')
        y_ = tf.placeholder(tf.float32, [None, 3], name='input_y')

    with tf.name_scope('hidden1'):
        with tf.name_scope('weights'):
            W1 = tf.Variable(tf.truncated_normal([241, hidden1_size], stddev=stddev), name='weights')
            variable_summaries(W1)
        with tf.name_scope('biases'):
            b1 = tf.Variable(tf.zeros([hidden1_size]), name='biases')
            variable_summaries(b1)
        with tf.name_scope('activation'):
            hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
            tf.summary.histogram('activation', hidden1)

    with tf.name_scope('hidden2'):
        with tf.name_scope('weights'):
            W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=stddev), name='weights')
            variable_summaries(W2)
        with tf.name_scope('biases'):
            b2 = tf.Variable(tf.zeros([hidden2_size]), name='biases')
            variable_summaries(b2)
        with tf.name_scope('activation'):
            hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
            tf.summary.histogram('activation', hidden2)


    with tf.name_scope('hidden3'):
        with tf.name_scope('weights'):
            W3 = tf.Variable(tf.truncated_normal([hidden2_size, hidden3_size], stddev=stddev), name='weights')
            variable_summaries(W3)
        with tf.name_scope('biases'):
            b3 = tf.Variable(tf.zeros([hidden3_size]), name='biases')
            variable_summaries(b3)
        with tf.name_scope('activation'):
            hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3)
            tf.summary.histogram('activation', hidden3)


    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        hidden3_drop = tf.nn.dropout(hidden3, keep_prob=keep_prob)

    with tf.name_scope('scores'):
        with tf.name_scope('weights'):
            W4 = tf.Variable(tf.truncated_normal([hidden3_size, 3], stddev=stddev), name='weights')
            variable_summaries(W4)
        with tf.name_scope('biases'):
            b4 = tf.Variable(tf.zeros([3]), name='biases')
            variable_summaries(b4)
        y = tf.matmul(hidden3_drop, W4) + b4

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name='xentropy')
        loss = (cross_entropy + reg * tf.nn.l2_loss(W1) + reg * tf.nn.l2_loss(b1) +
                reg * tf.nn.l2_loss(W2) + reg * tf.nn.l2_loss(b2) +
                reg * tf.nn.l2_loss(W3) + reg * tf.nn.l2_loss(b3) +
                reg * tf.nn.l2_loss(W4) + reg * tf.nn.l2_loss(b4))
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('loss', loss)
    if use_L2 == True:
        train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss)
    else:
        train_step = tf.train.AdamOptimizer(lr_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    return train_step, accuracy, loss, (x, y_, keep_prob), merged


#################################################################
################## train part ###################################
#################################################################
def do_eval(sess, data_set, batch_size, accuracy, placeholders, merged, test_writer, global_step, if_summary):
    x, y_, keep_prob = placeholders
    num_epoch = len(data_set) / batch_size
    reset_data_size = len(data_set) % batch_size

    index = 0
    count = 0.0
    for step in xrange(num_epoch):
        index, data = sequence_get_data(data_set, index, batch_size)
        if step == num_epoch - 1:
            if if_summary:
                summary, result = sess.run([merged, accuracy], feed_dict={x: data['features'], y_: data['labels'],
                                                                     keep_prob: 1.0})
                test_writer.add_summary(summary, global_step)
            else:
                result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob: 1.0})
        else:
            result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob:1.0})
        count += result * batch_size
    if reset_data_size != 0:
        #the reset data
        index, data = sequence_get_data(data_set, index, reset_data_size)
        result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob:1.0})
        count += result * reset_data_size
    return count / len(data_set)

def train(max_step, train_dataset, validation_dataset, test_dataset, batch_size, sess, keep_prob_v, loss, accuracy,
          train_step, placeholders, lr_rate, lr_decay, lr_decay_epoch, dir_path, merged):

    x, y_, keep_prob = placeholders

    train_path = 'tmp/train'
    test_path = 'tmp/test'

    if tf.gfile.Exists(train_path):
        tf.gfile.DeleteRecursively(train_path)
    tf.gfile.MakeDirs(train_path)
    if tf.gfile.Exists(test_path):
        tf.gfile.DeleteRecursively(test_path)
    tf.gfile.MakeDirs(test_path)

    train_writer = tf.summary.FileWriter(train_path, sess.graph)
    test_writer = tf.summary.FileWriter(test_path)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    index = 0
    last_accuracy = 0.0
    best_accuracy = 0.8
    best_path = ''
    # Train
    for step in range(max_step):
        #before_time = time.time()
        data = get_batch_data(train_dataset, batch_size)
        # data = get_whole_data(train_dataset)
        # index, data = sequence_get_data(train_dataset, index, batch)


        if step % 20 or step == max_step - 1:
            summary, _, loss_v = sess.run([merged, train_step, loss],
                             feed_dict={x: data['features'], y_: data['labels'], keep_prob: keep_prob_v})
            train_writer.add_summary(summary, step)
        else:
            _, loss_v = sess.run([train_step, loss],
                                 feed_dict={x: data['features'], y_: data['labels'], keep_prob: keep_prob_v})

        if step % 100 == 0:
            print 'loss in step %d is %f' % (step, loss_v)

            #last_time = time.time()
            #span_time = last_time - before_time
            #print ('100 steps is %f second' % (span_time * 100))
            #print ('rest time is %f minutes' % (span_time * (max_step - step) / 60))
        if step % 500 == 0 or step == max_step - 1:
            # data = get_batch_data(test_dataset, 1000)
            # result = sess.run(accuracy, feed_dict={x: data['features'], y_: data['labels'], keep_prob:1.0})
            result = do_eval(sess, train_dataset, batch_size, accuracy, placeholders, merged, test_writer, step, False)
            print '----------train acc in step %d is %f-------------' % (step, result)
            result = do_eval(sess, validation_dataset, batch_size, accuracy, placeholders, merged, test_writer, step,
                             True)
            print '----------accuracy in step %d is %f-------------' % (step, result)
            if result > last_accuracy:
                last_accuracy = result
                if last_accuracy > best_accuracy:
                    best_accuracy = result
                    path = "modules/%s/%.2f/model.ckpt" % (dir_path, result)
                    best_path = path
                    if tf.gfile.Exists(path):
                        tf.gfile.DeleteRecursively(path)
                    tf.gfile.MakeDirs(path)
                    save_path = saver.save(sess, path)
                    print("Model saved in file: %s" % save_path)


        if (step % lr_decay_epoch == 0 and step > 0):
            lr_rate *= lr_decay

    if best_path != '':
        saver.restore(sess, best_path)
        print "Model restored."

    result = do_eval(sess, test_dataset, batch_size, accuracy, placeholders, merged, test_writer, step, False)
    print '-----------last accuracy is %f------------' % (result)

    if result > 0.98:
        filename = 'modules/%f-%s' % (result, dir_path)
        f = file(filename, 'w+')
        f.write(dir_path)
        f.close()
        print 'best file writed'

        train_log_path = 'modules/%s/logs/train' % dir_path
        test_log_path = 'modules/%s/logs/test' % dir_path
        if tf.gfile.Exists(train_log_path):
            tf.gfile.DeleteRecursively(train_log_path)
        tf.gfile.MakeDirs(train_log_path)
        if tf.gfile.Exists(test_log_path):
            tf.gfile.DeleteRecursively(test_log_path)
        tf.gfile.MakeDirs(test_log_path)

        copyFiles(train_path, train_log_path)
        copyFiles(test_path, test_log_path)

    if result < 0.4:
        train_log_path = 'modules/low/%s/logs/train' % dir_path
        test_log_path = 'modules/low/%s/logs/test' % dir_path
        if tf.gfile.Exists(train_log_path):
            tf.gfile.DeleteRecursively(train_log_path)
        tf.gfile.MakeDirs(train_log_path)
        if tf.gfile.Exists(test_log_path):
            tf.gfile.DeleteRecursively(test_log_path)
        tf.gfile.MakeDirs(test_log_path)

        copyFiles(train_path, train_log_path)
        copyFiles(test_path, test_log_path)

        print 'saved the low accuracy log'

    train_writer.close()
    test_writer.close()

    return best_accuracy

#######################################################################
################# begin train #########################################
#######################################################################


filename = 'ciena_test.csv'

batch = 100
lr_rate = 0.01
max_step = 3500
reg = 0.02
lr_decay = 1.0
lr_decay_epoch = 800
keep_prob_v = 1.0
stddev = 0.8
use_L2 = False
dir_path = 'modules'
hidden1_size = 200
hidden2_size = 100
hidden3_size = 30


dataset = pd.read_csv(filename, header=None)
train_dataset, validation_dataset, test_dataset = split_dataset(dataset, radio=0.1)

loop = 2 ** 8
time_span = 0.0
for reg in [0.01, 0.02]:
    for lr_rate in [0.01, 0.05]:
        for stddev in [1.0, 0.35]:
            for lr_decay in [1.0, 0.99]:
                for hidden1_size in [200, 150]:
                    for hidden2_size in [120, 100]:
                        for hidden3_size in [50, 30]:
                            for use_L2 in [True, False]:
                                before_time = time.time()
                                print '-------------------now changed-----------------'
                                print 'reg is', reg
                                print 'lr_rate is', lr_rate
                                print 'stddev is', stddev
                                print 'lr_decay', lr_decay
                                print 'hidden1 size is', hidden1_size
                                print 'hidden2 size is', hidden2_size
                                print 'hidden3 size is', hidden3_size
                                print 'use L2 is', use_L2
                                print 'rest loop is', loop
                                print 'last use time %.2f' % time_span
                                print 'rest time is %.2f minute' % (time_span * loop / 60)
                                print '------------------------------------------------'

                                with tf.Graph().as_default():
                                    with tf.Session() as sess:
                                        train_step, accuracy, loss, placeholders, merged = inference(hidden1_size,hidden2_size,hidden3_size,lr_rate, reg,stddev,use_L2=use_L2)

                                        dir_path = "r%.2flr%.2fs%.2fld%.2fh%d%d%du%s" % (
                                        reg, lr_rate, stddev, lr_decay,
                                        hidden1_size, hidden2_size, hidden3_size, use_L2)

                                        best_accuracy = train(max_step, train_dataset, validation_dataset, test_dataset,
                                                              batch, sess, keep_prob_v, loss, accuracy, train_step,
                                                              placeholders, lr_rate, lr_decay, lr_decay_epoch,
                                                              dir_path, merged)

                                        if best_accuracy < 0.8:
                                            train(max_step, train_dataset, validation_dataset, test_dataset,
                                                  batch, sess, keep_prob_v, loss, accuracy, train_step, placeholders,
                                                  lr_rate, lr_decay, lr_decay_epoch, dir_path, merged)

                                after_time = time.time()
                                time_span = after_time - before_time
                                loop -= 1


""""""
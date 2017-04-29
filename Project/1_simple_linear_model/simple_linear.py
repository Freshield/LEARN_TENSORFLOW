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
    columns = data_set.values[random_index]
    features = columns[:, :-20]
    labels = columns[:, -1]
    return {'features': features, 'labels': labels}


def get_whole_data(data_set):
    features = data_set.values[:, :-20]
    labels = data_set.values[:, -1]
    return {'features': features, 'labels': labels}

def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    # index = tf.random_shuffle(tf.range(0, data_size))
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs


def sequence_get_data(data_set, indexs, last_index, batch_size):
    next_index = last_index + batch_size
    out_of_dataset = False

    if next_index > data_set.shape[0]:

        next_index -= data_set.shape[0]
        last_part = np.arange(last_index,indexs.shape[0])
        before_part = np.arange(next_index)
        span_index = indexs[np.concatenate((last_part, before_part))]
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]

    columns = data_set.values[span_index]
    features = columns[:, :-20]
    labels = columns[:, -1]
    return (next_index, {'features': features, 'labels': labels}, out_of_dataset)

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

def weight_variable(shape, stddev, summary=True):
    with tf.name_scope('weights'):
        weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev), name='weights')
        if summary == True:
            variable_summaries(weights)
    return weights

def biases_variable(shape, value, summary=True):
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.constant(value=value, dtype=tf.float32, shape=shape, name='biases'))
        if summary == True:
            variable_summaries(biases)
    return biases

def get_hidden(input, input_size, hidden_size, stddev, b_value, name='hidden', act=tf.nn.relu, summary = True):
    with tf.name_scope(name):
        W = weight_variable([input_size, hidden_size], stddev, summary)
        b = biases_variable([hidden_size], b_value, summary)
        with tf.name_scope('activation'):
            hidden = act(tf.matmul(input, W) + b)
            if summary == True:
                tf.summary.histogram('activation', hidden)

        l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

    return hidden, l2_loss

def get_droppout(input):
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        dropout = tf.nn.dropout(input, keep_prob=keep_prob)
    return dropout, keep_prob

def get_inputs():
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 241], name='input_x')
        y_ = tf.placeholder(tf.int32, [None], name='input_y')
        y_one_hot = tf.one_hot(y_, 3)

    return x, y_, y_one_hot

def get_logits(x, hidden1_size, hidden2_size, hidden3_size, labels_size, stddev, b_value):

    hidden1, h1_l2_loss = get_hidden(x, 241, hidden1_size, stddev, b_value, 'hidden1')

    hidden2, h2_l2_loss = get_hidden(hidden1, hidden1_size, hidden2_size, stddev, b_value, 'hidden2')

    hidden2_drop, keep_prob = get_droppout(hidden2)

    hidden3, h3_l2_loss = get_hidden(hidden2_drop, hidden2_size, hidden3_size, stddev, b_value, 'hidden3')

    y, h4_l2_loss = get_hidden(hidden3, hidden3_size, labels_size, stddev, b_value, 'scores')

    l2_loss = h1_l2_loss + h2_l2_loss + h3_l2_loss + h4_l2_loss

    return y, l2_loss, keep_prob

def get_loss(y, y_one_hot, l2_loss, reg, summary = True, use_l2_loss = True):
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=y), name='xentropy')
        if use_l2_loss == True:
            loss = cross_entropy + 0.5 * reg * l2_loss
        else:
            loss = cross_entropy

        if summary == True:
            tf.summary.scalar('cross_entropy', cross_entropy)
            tf.summary.scalar('loss', loss)

        return loss

def get_train_op(loss, lr_rate, optimizer=tf.train.AdamOptimizer):
    with tf.name_scope('train'):
        train_op = optimizer(lr_rate).minimize(loss)
    return train_op

def get_correct_num(y, y_one_hot):
    with tf.name_scope('correct_num'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_one_hot, 1))
        correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    return correct_num

def get_accuracy(y, y_one_hot):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def get_feed_dict(placeholders, data, keep_prob_v):
    x, y_, keep_prob = placeholders
    return {x: data['features'], y_: data['labels'], keep_prob:keep_prob_v}

def do_batch_eval(sess, data_set, batch_size, accuracy, placeholders,
                  merged, test_writer, global_step,if_summary=True):
    indexs = get_random_seq_indexs(data_set)
    last_index = 0
    _, data, _ = sequence_get_data(data_set, indexs, last_index, batch_size)

    feed_dict = get_feed_dict(placeholders, data, 1.0)

    if if_summary == True:
        summary, result = sess.run([merged, accuracy], feed_dict=feed_dict)
        test_writer.add_summary(summary, global_step)
    else:
        result = sess.run(accuracy, feed_dict=feed_dict)

    return result

def do_eval(sess, data_set, correct_num, placeholders):
    batch_size = 100
    num_epoch = len(data_set) / batch_size
    reset_data_size = len(data_set) % batch_size

    indexs = get_random_seq_indexs(data_set)
    last_index = 0
    count = 0
    for step in xrange(num_epoch):
        last_index, data, _ = sequence_get_data(data_set, indexs, last_index, batch_size)
        feed_dict = get_feed_dict(placeholders, data, 1.0)
        num = sess.run(correct_num, feed_dict=feed_dict)
        count += num

    if reset_data_size != 0:
        #the reset data
        last_index, data, _ = sequence_get_data(data_set, indexs, last_index, reset_data_size)
        feed_dict = get_feed_dict(placeholders, data, 1.0)
        num = sess.run(correct_num, feed_dict=feed_dict)
        count += num
    return count / data_set.shape[0]

def del_and_create_dir(dir_path):
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)
    tf.gfile.MakeDirs(dir_path)

def store_model(last_accuracy, best_accuracy, best_path, dir_path, saver, sess, step):
    if last_accuracy > best_accuracy or last_accuracy == 1.0:
        best_accuracy = last_accuracy
        path = "modules/%s/%d/%.2f/model.ckpt" % (dir_path, step, last_accuracy)
        best_path = path
        del_and_create_dir(path)
        save_path = saver.save(sess, path)
        print("Model saved in file: %s" % save_path)
    return best_path, best_accuracy

def write_file(result, dir_path, situation_now):
    filename = 'modules/%f-%s' % (result, dir_path)
    f = file(filename, 'w+')
    f.write(dir_path)
    f.write(situation_now)
    f.close()
    print 'best file writed'



def train(max_step, datasets, batch_size, sess, keep_prob_v, loss, accuracy,
          train_op, placeholders, lr_rate, lr_decay, lr_decay_epoch, correct_num,
          dir_path, merged, situation_now, loop):
    train_dataset, validation_dataset, test_dataset = datasets

    train_path = 'tmp/train'
    test_path = 'tmp/test'

    del_and_create_dir(train_path)
    del_and_create_dir(test_path)

    train_writer = tf.summary.FileWriter(train_path, sess.graph)
    test_writer = tf.summary.FileWriter(test_path)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    last_accuracy = 0.0
    best_accuracy = 0.8
    best_path = ''

    indexs = get_random_seq_indexs(train_dataset)
    out_of_dataset = False
    last_index = 0
    saver = tf.train.Saver()
    train_acc = 0
    # Train
    for step in xrange(max_step):
        before_time = time.time()
        if out_of_dataset == True:
            indexs = get_random_seq_indexs(train_dataset)
            last_index = 0
            out_of_dataset = False

        last_index, data, out_of_dataset = sequence_get_data(train_dataset, indexs, last_index, batch_size)

        feed_dict = get_feed_dict(placeholders, data, keep_prob_v)
        #write summary
        if step % 20 or step == max_step - 1:
            summary, _, loss_v = sess.run([merged, train_op, loss],feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
        else:
            _, loss_v = sess.run([train_op, loss],feed_dict=feed_dict)
        #write loss and time
        if step % 100 == 0:
            print '-----------loss in step %d is %f----------' % (step, loss_v)

            last_time = time.time()
            span_time = last_time - before_time
            print ('100 steps is %f second' % (span_time * 100))
            print ('rest time is %f minutes' % (span_time * (max_step - step) * loop / 60))
        #do evaluation
        if step % 500 == 0 or step == max_step - 1:

            result = do_batch_eval(
                sess, train_dataset, 1000, accuracy, placeholders,
                merged, test_writer, step, True)
            print '----------train acc in step %d is %f-------------' % (step, result)
            result = do_batch_eval(
                sess, validation_dataset, 1000, accuracy, placeholders,
                merged, test_writer, step, True)
            print '----------valid acc in step %d is %f-------------' % (step, result)
            if result > last_accuracy:
                last_accuracy = result
                best_path, best_accuracy = store_model(last_accuracy, best_accuracy, best_path, dir_path, saver, sess, step)

        if step % 10000 == 0:
            train_acc = do_eval(sess, train_dataset, correct_num, placeholders)
            result = do_eval(sess, validation_dataset, correct_num, placeholders)
            print '----------valid whole acc in step %d is %f-------------' % (step, result)

        if (step % lr_decay_epoch == 0 and step > 0):
            lr_rate *= lr_decay

    if best_path != '':
        saver.restore(sess, best_path)
        print "Model restored."

    result = do_eval(sess, validation_dataset, correct_num, placeholders)
    print '-----------last accuracy is %f------------' % (result)

    if result > 0.98:
        write_file(result, dir_path, situation_now)
        write_file(train_acc, dir_path, situation_now)

        train_log_path = 'modules/%s/logs/train' % dir_path
        test_log_path = 'modules/%s/logs/test' % dir_path
        del_and_create_dir(train_log_path)
        del_and_create_dir(test_log_path)
        copyFiles(train_path, train_log_path)
        copyFiles(test_path, test_log_path)

    if result < 0.4:
        train_log_path = 'modules/low/%s/logs/train' % dir_path
        test_log_path = 'modules/low/%s/logs/test' % dir_path
        del_and_create_dir(train_log_path)
        del_and_create_dir(test_log_path)
        copyFiles(train_path, train_log_path)
        copyFiles(test_path, test_log_path)

        print 'saved the low accuracy log'

    train_writer.close()
    test_writer.close()

    return best_accuracy
import tensorflow as tf
import pandas as pd
import numpy as np
import time

def split_dataset(dataset, radio):
    test_dataset_size = int(radio * len(dataset))
    data_size = dataset.shape[0]
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    train_set = dataset.values[indexs[:-test_dataset_size * 2]]
    validation_set = dataset.values[indexs[-test_dataset_size * 2 : -test_dataset_size]]
    test_set = dataset.values[indexs[-test_dataset_size : ]]
    return train_set, validation_set, test_set

def normalize_dataset(dataset, min_values=None, max_values=None):
    norm_dataset = np.zeros((dataset.shape))
    norm_dataset[:, :] = dataset[:, :]

    if min_values == None:
        CM_min = np.min(norm_dataset[:,:200])
        CD_min = np.min(norm_dataset[:,200:201])
        length_min = np.min(norm_dataset[:,201:221])
        power_min = np.min(norm_dataset[:,221:241])
    else:
        CM_min, CD_min, length_min, power_min = min_values


    if max_values == None:
        CM_max = np.max(norm_dataset[:,0:200])
        CD_max = np.max(norm_dataset[:,200:201])
        length_max = np.max(norm_dataset[:,201:221])
        power_max = np.max(norm_dataset[:,221:241])
    else:
        CM_max, CD_max, length_max, power_max = max_values

    def calcul_norm(dataset, min, max):
        return (dataset - min) / (max - min)


    norm_dataset[:,0:200] = calcul_norm(norm_dataset[:,0:200], CM_min, CM_max)
    norm_dataset[:,200:201] = calcul_norm(norm_dataset[:,200:201], CD_min, CD_max)
    norm_dataset[:,201:221] = calcul_norm(norm_dataset[:,201:221], length_min, length_max)
    norm_dataset[:,221:241] = calcul_norm(norm_dataset[:,221:241], power_min, power_max)

    min_values = (CM_min, CD_min, length_min, power_min)
    max_values = (CM_max, CD_max, length_max, power_max)

    return norm_dataset, min_values, max_values

def get_batch_data(data_set, batch_size):
    random_index = np.random.randint(data_set.shape[0], size=[batch_size])
    columns = data_set[random_index]
    features = columns[:,:241]
    labels = columns[:,-10]
    return {'features':features, 'labels':labels}

def get_whole_data(data_set):
    features = data_set[:,:241]
    labels = data_set[:,-10]
    return {'features':features, 'labels':labels}


def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  weight = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
  #weight = tf.Variable(np.random.randn(shape[0],shape[1]).astype(np.float32) * np.sqrt(2.0/shape[0]),)
  return weight

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

"""
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
"""

def batch_norm_layer1(x, train_phase, scope_bn):
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = np.arange(len(x.shape) - 1)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

#ensure the path exist
def del_and_create_dir(dir_path):
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)
    tf.gfile.MakeDirs(dir_path)

def random_uniform_array(number, start, end):
    array = np.zeros(number)
    for i in np.arange(number):
        array[i] = 10 ** np.random.uniform(start, end)

    return array

#-----------------------------
#filename='ciena_test.csv'
filename = 'pca1000_1.csv'
#filename = 'norm.csv'
dataset = pd.read_csv(filename, header=None)
train_dataset, validation_dataset, test_dataset = split_dataset(dataset, 0.1)
train_dataset, train_mins, train_maxs = normalize_dataset(train_dataset)
validation_dataset,_,_ = normalize_dataset(validation_dataset,train_mins, train_maxs)
test_dataset,_,_ = normalize_dataset(test_dataset,train_mins, train_maxs)

regs = random_uniform_array(20, -5, -1)
lr_rates = random_uniform_array(20, -7, -2)
keeps = random_uniform_array(10, -0.3, 1)
max_step = 15000

log = ''
count = len(regs) * len(lr_rates) * len(keeps)

for reg in regs:
    for lr_rate in lr_rates:
        for keep in keeps:
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    info = '--------------------------------------\n' \
                           '-------reg is %f\n' \
                           '-------lr_rate is %f\n' \
                           '-------keep is %f\n' \
                           '--------------------------------------' % (reg, lr_rate, keep)
                    print info
                    log += info + '\n'
                    x = tf.placeholder(tf.float32, [None, 241])
                    y_ = tf.placeholder(tf.int32, [None])
                    y_one_hot = tf.one_hot(y_, 3)
                    train_phase = tf.placeholder(tf.bool, name='train_phase')
                    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

                    bn_input = batch_norm_layer1(x, train_phase, 'bn_input')

                    W1 = weight_variable([241, 512], 'W1')
                    variable_summaries(W1)
                    b1 = tf.Variable(tf.constant(0.1, shape=[512]))
                    h1 = tf.matmul(bn_input, W1) + b1
                    bn_h1 = batch_norm_layer1(h1, train_phase, 'bn_h1')
                    act_h1 = tf.nn.relu(bn_h1)
                    tf.summary.histogram('act_h1', act_h1)
                    drop_h1 = tf.nn.dropout(act_h1, keep_prob=keep_prob)

                    W2 = weight_variable([512, 256], 'W2')
                    variable_summaries(W2)
                    b2 = tf.Variable(tf.constant(0.1, shape=[256]))
                    h2 = tf.matmul(drop_h1, W2) + b2
                    bn_h2 = batch_norm_layer1(h2, train_phase, 'bn_h2')
                    act_h2 = tf.nn.relu(bn_h2)
                    tf.summary.histogram('act_h2', act_h2)
                    drop_h2 = tf.nn.dropout(act_h2, keep_prob=keep_prob)

                    W3 = weight_variable([256, 128], 'W3')
                    variable_summaries(W3)
                    b3 = tf.Variable(tf.constant(0.1, shape=[128]))
                    h3 = tf.matmul(drop_h2, W3) + b3
                    bn_h3 = batch_norm_layer1(h3, train_phase, 'bn_h3')
                    act_h3 = tf.nn.relu(bn_h3)
                    tf.summary.histogram('act_h3', act_h3)
                    drop_h3 = tf.nn.dropout(act_h3, keep_prob=keep_prob)

                    W4 = weight_variable([128, 64], 'W4')
                    variable_summaries(W4)
                    b4 = tf.Variable(tf.constant(0.1, shape=[64]))
                    h4 = tf.matmul(drop_h3, W4) + b4
                    bn_h4 = batch_norm_layer1(h4, train_phase, 'bn_h4')
                    act_h4 = tf.nn.relu(bn_h4)
                    tf.summary.histogram('act_h4', act_h4)
                    drop_h4 = tf.nn.dropout(act_h4, keep_prob=keep_prob)

                    W5 = weight_variable([64, 3], 'W5')
                    variable_summaries(W5)
                    b5 = tf.Variable(tf.constant(0.1, shape=[3]))
                    y = tf.matmul(drop_h4, W5) + b5

                    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=y))

                    reg_loss = 0.5 * reg * (
                        tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(
                            W3) + tf.nn.l2_loss(b3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(b4) + tf.nn.l2_loss(
                            W5) + tf.nn.l2_loss(b5))

                    loss = cross_entropy + reg_loss
                    tf.summary.scalar('loss', loss)

                    train_op = tf.train.AdamOptimizer(lr_rate).minimize(loss)

                    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_one_hot, 1))

                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    tf.summary.scalar('acc', accuracy)

                    merged = tf.summary.merge_all()

                    # define store path
                    train_path = 'tmp/train'
                    test_path = 'tmp/test'
                    del_and_create_dir(train_path)
                    del_and_create_dir(test_path)

                    # create writer for tensorboard
                    train_writer = tf.summary.FileWriter(train_path, sess.graph)
                    test_writer = tf.summary.FileWriter(test_path)

                    sess.run(tf.global_variables_initializer())

                    for step in xrange(max_step):
                        before_time = time.time()

                        data = get_batch_data(train_dataset, 100)

                        feed_dict = {x: data['features'], y_: data['labels'], train_phase: True, keep_prob: keep}

                        if step % 40 == 0 or step == max_step - 1:
                            summary, _, loss_v, acc = sess.run([merged, train_op, loss, accuracy], feed_dict=feed_dict)
                            train_writer.add_summary(summary, step)
                        else:
                            _, loss_v, acc = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)

                        if step % 100 == 0:
                            print '---loss in step %d is %f, acc is %.3f---' % (step, loss_v, acc)
                            log += '---loss in step %d is %f, acc is %.3f---\n' % (step, loss_v, acc)

                        if step % 500 == 0:
                            last_time = time.time()
                            span_time = last_time - before_time
                            print ('last 500 loop use %f minutes' % (span_time * 500 / 60))
                            print ('rest time is %.3f minutes' % (span_time * (max_step - step) * count / 60))
                            log += ('last 500 loop use %f minutes\n' % (span_time * 500 / 60))
                            log += ('rest time is %.3f minutes\n' % (span_time * (max_step - step) * count / 60))
                            data = get_whole_data(train_dataset)

                            feed_dict = {x: data['features'], y_: data['labels'], train_phase: False, keep_prob: 1.0}

                            result = sess.run(accuracy, feed_dict=feed_dict)

                            print '--------train acc in step %d is %f--------' % (step, acc)
                            log += '--------train acc in step %d is %f--------\n' % (step, acc)

                            data = get_whole_data(validation_dataset)

                            feed_dict = {x: data['features'], y_: data['labels'], train_phase: False, keep_prob: 1.0}

                            summary, result = sess.run([merged, accuracy], feed_dict=feed_dict)
                            test_writer.add_summary(summary, step)

                            print '--------valida acc in step %d is %f--------' % (step, result)
                            log += '--------valida acc in step %d is %f--------\n' % (step, result)

                        if step == max_step - 1:
                            data = get_whole_data(test_dataset)

                            feed_dict = {x: data['features'], y_: data['labels'], train_phase: False, keep_prob: 1.0}

                            result = sess.run(accuracy, feed_dict=feed_dict)

                            print '--------test acc in step %d is %f--------' % (step, result)
                            log += '--------test acc in step %d is %f--------\n' % (step, result)

                        if step % 1000 == 0:
                            lr_rate = lr_rate * 0.99

                    test_writer.close()
                    train_writer.close()

                    del_and_create_dir('log')
                    filename = 'log/%.3f_loop%d' % (result, count)
                    f = file(filename, 'w+')
                    f.write(log)
                    f.close()

                    count -= 1





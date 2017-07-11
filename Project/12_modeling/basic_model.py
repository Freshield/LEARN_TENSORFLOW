import tensorflow as tf
import numpy as np
import time


from data_process_model import *
from file_system_model import *

#show the processing
#ver 1.0
process_line = {
    10:"[==========>]",
    9: "[=========> ]",
    8: "[========>  ]",
    7: "[=======>   ]",
    6: "[======>    ]",
    5: "[=====>     ]",
    4: "[====>      ]",
    3: "[===>       ]",
    2: "[==>        ]",
    1: "[=>         ]",
    0: "[>          ]"
}

#create weights
#ver 1.0
def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  weight = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

  return weight


#create biases
#ver 1.0
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

#for create convolution kernel
#ver 1.0
def conv2d(x, W, stride, padding):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


# for create batch norm layer
#ver 1.0
def batch_norm_layer(x, train_phase, scope_bn):
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

#dense layer
#ver 1.0
def dense_layer(input_layer, output_size, name, act=tf.nn.relu):
    with tf.variable_scope(name):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, output_size], 'dense_weight')
        b = bias_variable([output_size])
        output = act(tf.matmul(input_layer, W) + b)
    return output, [W]

# score layer
#ver 1.0
def score_layer(input_layer, label_size):
    with tf.variable_scope("score"):
        input_size = input_layer.shape[-1]
        W = weight_variable([input_size, label_size], 'score_weight')
        b = bias_variable([label_size])
        output = tf.matmul(input_layer, W) + b
    return output, [W]

#get the correct number and accuracy
#ver 1.0
def corr_num_acc(labels, logits):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_num, accuracy

#get the loss
#ver 1.0
def loss(labels, logits, reg=None, parameters=None):
    with tf.variable_scope("loss"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy'))

        if parameters == None:
            cost = cross_entropy
        else:
            reg_loss = 0.0
            for para in parameters:
                reg_loss += reg * 0.5 * tf.nn.l2_loss(para)
            cost = cross_entropy + reg_loss
    return cost

# to do the evaluation part for the whole data
# not use all data together, but many batchs
#ver 1.0
def do_eval(sess, X_dataset, para_dataset, y_dataset, batch_size, correct_num, placeholders, merged=None, test_writer=None,
            global_step=None):
    # get the placeholders
    input_x, para_pl, input_y, train_phase, keep_prob = placeholders
    # calculate the epoch and rest data
    num_epoch = X_dataset.shape[0] // batch_size
    rest_data_size = X_dataset.shape[0] % batch_size

    index = 0
    count = 0.0
    indexs = np.arange(X_dataset.shape[0])

    for step in xrange(num_epoch):
        index, data, _ = sequence_get_data(X_dataset, para_dataset, y_dataset, indexs, index, batch_size)

        feed_dict = {input_x: data['X'], para_pl:data['p'], input_y: data['y'], train_phase: False, keep_prob: 1.0}

        if step != num_epoch - 1 or merged == None:
            num = sess.run(correct_num, feed_dict=feed_dict)
        else:
            summary, num = sess.run([merged, correct_num], feed_dict=feed_dict)
            # test_writer.add_summary(summary, global_step)

        count += num

    if rest_data_size != 0:
        # the rest data
        index, data, _ = sequence_get_data(X_dataset, y_dataset, indexs, index, rest_data_size)

        feed_dict = {input_x: data['X'], para_pl:data['p'], input_y: data['y'], train_phase: False, keep_prob: 1.0}

        num = sess.run(correct_num, feed_dict=feed_dict)

        count += num
    return count / X_dataset.shape[0]

#train loop in one file
#ver 1.0
def do_train_file(sess, placeholders, dir, train_file, SPAN, max_step, batch_size, keep_prob_v):
    input_x, para_pl, input_y, train_phase, keep_prob, train_step, loss_value, accuracy = placeholders

    X_train, para_train, y_train = prepare_dataset(dir, train_file, SPAN)

    indexs = get_random_seq_indexs(X_train)
    out_of_dataset = False
    last_index = 0

    loop_loss_v = 0.0
    loop_acc = 0.0

    # one loop, namely, one file
    for step in xrange(max_step):

        # should not happen
        if out_of_dataset == True:
            print "out of dataset"
            indexs = get_random_seq_indexs(X_train)
            last_index = 0
            out_of_dataset = False

        last_index, data, out_of_dataset = sequence_get_data(X_train, para_train, y_train, indexs, last_index,batch_size)

        feed_dict = {input_x: data['X'], para_pl:data['p'], input_y: data['y'], train_phase: True, keep_prob: keep_prob_v}
        _, loss_v, acc = sess.run([train_step, loss_value, accuracy], feed_dict=feed_dict)

        loop_loss_v += loss_v
        loop_acc += acc

    loop_loss_v /= max_step
    loop_acc /= max_step
    return loop_loss_v, loop_acc

#to show the time
#ver 1.0
def time_show(before_time, last_loop_num, loop_now, total_loop, epoch_now, total_epoch, log, count = None, count_total = None):
    last_time = time.time()
    span_time = last_time - before_time
    rest_loop = total_loop - loop_now
    rest_epoch = total_epoch - epoch_now

    #show the last loop time
    words = 'last %d loop use %f minutes' % (last_loop_num, span_time * last_loop_num / 60)
    words_log_print(words, log)

    #show the rest loop time
    words = 'rest loop need %.3f minutes' % (span_time * rest_loop / 60)
    words_log_print(words, log)

    #show the rest epoch time
    words = 'rest epoch need %.3f hours' % (span_time * rest_loop / 3600 + span_time * total_loop * rest_epoch / 3600)
    words_log_print(words, log)

    # show the cross valid total time
    if count != None:
        rest_count = count_total - count
        words = 'rest total time need %.3f hours' % (span_time * rest_loop / 3600 + span_time * total_loop * rest_epoch / 3600 + span_time * total_loop * total_epoch * rest_count / 3600)
        words_log_print(words, log)


#to get the random hypers for cross valid
#ver 1.0
def random_uniform_array(number, start, end):
    array = np.zeros(number)
    for i in np.arange(number - 2):
        array[i] = 10 ** np.random.uniform(start, end)
    array[-2] = 10 ** start
    array[-1] = 10 ** end

    return array

#log add word and print it
#ver 1.0
def words_log_print(words, log):
    print words
    log += words + "\n"

#show the words and add to log for epoch
#ver 1.0
def words_log_print_epoch(epoch, epochs, log ):
    words = "\nepoch "
    words += process_line[int(10 * (float(epoch) / float(epochs)))]
    words += "[%d/%d]\n" % (epoch, epochs)
    words_log_print(words, log)

#show the words and add to log for loop
#ver 1.0
def words_log_print_loop(loop, loops, loop_loss_v, loop_acc, log ):
    words = "loop "
    words += process_line[int(10 * (float(loop) / float(loops)))]
    words += "[%d/%d] " % (loop, loops)
    words += 'loss in loop %d is %f, acc is %.3f' % (loop, loop_loss_v, loop_acc)
    words_log_print(words, log)

# do the evaluation for the last x files
#ver 1.0
def evaluate_last_x_files(number, eval_parameters):

    loop, loop_indexs, SPAN, sess, batch_size, correct_num, placeholders, log = eval_parameters

    train_acc = 0.0
    valid_acc = 0.0
    print "step",
    for step in xrange(number):
        print step,
        # careful for the file name
        train_file = "train_set_%d.csv" % loop_indexs[loop - 10 + step]
        validation_file = "validation_set_%d.csv" % loop_indexs[loop - 10 + step]

        X_train, para_train, y_train = prepare_dataset(dir, train_file, SPAN)
        X_valid, para_valid, y_valid = prepare_dataset(dir, validation_file, SPAN)

        step_train_acc = do_eval(sess, X_train, para_train, y_train, batch_size, correct_num, placeholders)
        train_acc += step_train_acc

        step_valid_acc = do_eval(sess, X_valid, para_valid, y_valid, batch_size, correct_num, placeholders)
        valid_acc += step_valid_acc

    train_acc /= 10
    valid_acc /= 10
    print ""
    words = '----------train acc in loop %d is %.4f----------' % (loop, train_acc)
    words_log_print(words, log)
    words = '----------valid acc in loop %d is %.4f----------' % (loop, valid_acc)
    words_log_print(words, log)

#do the evalute for all of the test files
#ver 1.0
def evaluate_test(test_parameter):
    loops, epoch, SPAN, sess, batch_size, correct_num, placeholders, log = test_parameter
    # each epoch do a test evaluation
    test_acc = 0.0
    print "step",
    for test_loop in xrange(loops):
        print test_loop,
        test_file = "test_set_%d.csv" % test_loop
        X_test, para_test, y_test = prepare_dataset(dir, test_file, SPAN)
        loop_test_acc = do_eval(sess, X_test, para_test, y_test, batch_size, correct_num, placeholders)
        test_acc += loop_test_acc
    test_acc /= loops
    print ""
    words = '----------epoch %d test accuracy is %f----------' % (epoch, test_acc)
    words_log_print(words, log)
    return test_acc

#store the log file
#ver 1.0
def store_log(log_dir, test_acc, epoch, log):
    filename = log_dir + '%.4f_epoch%d' % (test_acc, epoch)
    f = file(filename, 'w+')
    f.write(log)
    f.close()

#store the module
#ver 1.0
def store_module(module_dir, test_acc, epoch, sess, log):
    saver = tf.train.Saver()
    module_path = module_dir + "%.4f_epoch%d/" % (test_acc, epoch)
    module_name = module_path + "module.ckpt"
    del_and_create_dir(module_path)
    save_path = saver.save(sess, module_name)
    words = "Model saved in file: %s" % save_path
    words_log_print(words, log)
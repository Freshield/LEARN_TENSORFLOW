import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os

################################################
#                                              #
#             lot of help function             #
#          create different network model      #
#                                              #
################################################

############################################################
############# helpers ######################################
############################################################
#to copy files from source dir to target dir
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

#split the dataset into three part:
#training, validation, test
def split_dataset(dataset, test_dataset_size=None, radio=None):
    if radio != None:
        test_dataset_size = int(radio * len(dataset))

    train_set = dataset[0:-test_dataset_size * 2]
    validation_set = dataset[-test_dataset_size * 2:-test_dataset_size]
    test_set = dataset[-test_dataset_size:len(dataset)]
    return train_set, validation_set, test_set

#get a random data(maybe have same value)
def get_batch_data(data_set, batch_size, spannum=20):
    lines_num = data_set.shape[0] - 1
    random_index = np.random.randint(lines_num, size=[batch_size])
    columns = data_set.values[random_index]
    label_num = spannum - 21
    features = columns[:, :-20]
    labels = columns[:, label_num]
    return {'features': features, 'labels': labels}

#directly get whole dataset(only for small dataset)
def get_whole_data(data_set, spannum=20):
    features = data_set.values[:, :-20]
    label_num = spannum - 21
    labels = data_set.values[:, label_num]
    return {'features': features, 'labels': labels}

#get a random indexs for dataset,
#so that we can shuffle the data every epoch
def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs

#use the indexs together,
#so that we can sequence batch whole dataset
def sequence_get_data(data_set, indexs, last_index, batch_size, spannum=20):
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
    label_num = spannum - 21
    labels = columns[:, label_num]
    return (next_index, {'features': features, 'labels': labels}, out_of_dataset)

#ensure the path exist
def del_and_create_dir(dir_path):
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)
    tf.gfile.MakeDirs(dir_path)

#write log file
def write_file(result, dir_path, situation_now):
    filename = 'modules/%f-%s' % (result, dir_path)
    f = file(filename, 'w+')
    f.write(dir_path)
    f.write(situation_now)
    f.close()
    print 'best file writed'

#####################################################################
############### create the graph ####################################
#####################################################################
#for summary the tensors
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

#create weights
#important
#the weight initial value stddev is kinds of hyper para
#if a wrong stddev will stuck the network
def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape=shape, stddev=stddev)

    return tf.Variable(initial, name='weights')

#create biases
def biases_variable(shape, value):
    initial = tf.constant(value=value, dtype=tf.float32, shape=shape)

    return tf.Variable(initial, name='biases')

#create hidden layer,
# relu(x * W + b)
#return parameter for L2 regularzation
def get_hidden(input, input_size, hidden_size, stddev, b_value, name='hidden', act=tf.nn.relu, summary = True):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W = weight_variable([input_size, hidden_size], stddev)
            if summary == True:
                variable_summaries(W)
        with tf.name_scope('biases'):
            b = biases_variable([hidden_size], b_value)
            if summary == True:
                variable_summaries(b)
        with tf.name_scope('activation'):
            activation = act(tf.matmul(input, W) + b, name='activation')
            if summary == True:
                tf.summary.histogram('activation', activation)

        parameters = (W, b)

    return activation, parameters

#drop the input
def get_droppout(input):
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        dropout = tf.nn.dropout(input, keep_prob=keep_prob)
    return dropout, keep_prob

#get input placeholders,
#also create one hot label here
def get_inputs():
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 241], name='input_x')
        y_ = tf.placeholder(tf.int32, [None], name='input_y')
        y_one_hot = tf.one_hot(y_, 3)

    return x, y_, y_one_hot

#important
#scores must different with hidden layer
#because there have relu at the end of hidden
#and score needn't
# x * W + b
#return parameter for L2 regularzation
def get_scores(input, input_size, hidden_size, stddev, b_value, name='scores', summary = True):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            W = weight_variable([input_size, hidden_size], stddev)
            if summary == True:
                variable_summaries(W)
        with tf.name_scope('biases'):
            b = biases_variable([hidden_size], b_value)
            if summary == True:
                variable_summaries(b)
        with tf.name_scope('scores'):
            y = tf.matmul(input, W) + b
            if summary == True:
                tf.summary.histogram('scores', y)

        parameters = (W, b)

    return y, parameters

#3 layers neural network model
#input 241, output 3
def get_logits(x, hidden1_size, hidden2_size, hidden3_size, labels_size, stddev, b_value):

    hidden1, h1_para = get_hidden(x, 241, hidden1_size, stddev, b_value, 'hidden1')

    hidden2, h2_para = get_hidden(hidden1, hidden1_size, hidden2_size, stddev, b_value, 'hidden2')

    hidden2_drop, keep_prob = get_droppout(hidden2)

    hidden3, h3_para = get_hidden(hidden2_drop, hidden2_size, hidden3_size, stddev, b_value, 'hidden3')

    y, h4_para = get_scores(hidden3, hidden3_size, labels_size, stddev, stddev)

    total_parameters = {'hidden1':h1_para,
                        'hidden2':h2_para,
                        'hidden3':h3_para,
                        'hidden4':h4_para}

    return y, total_parameters, keep_prob

#get loss by softmax
#also can choose if use L2 regularzation or not
def get_loss(y, y_one_hot, total_parameters, reg, summary = True, use_l2_loss = True):
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=y), name='xentropy')
        if use_l2_loss == True:
            reg_loss = 0
            for i in range(4):
                W, b = total_parameters['hidden'+str(i+1)]
                reg_loss += 0.5 * reg * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))
            loss = cross_entropy + reg_loss
        else:
            loss = cross_entropy

        if summary == True:
            tf.summary.scalar('cross_entropy', cross_entropy)
            tf.summary.scalar('loss', loss)

        return loss

#get train handle for training and backpropagation
#use adam function
def get_train_op(loss, lr_rate, optimizer=tf.train.AdamOptimizer):
    with tf.name_scope('train'):
        train_op = optimizer(lr_rate).minimize(loss)
    return train_op

#get the correct numbers
#so that we can get batch data correct number
#and add them together at the last to get the total accuracy
def get_correct_num(y, y_one_hot):
    with tf.name_scope('correct_num'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_one_hot, 1))
        correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    return correct_num

#get the feed data accuracy
def get_accuracy(y, y_one_hot):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

#get the feed dictionary
def get_feed_dict(placeholders, data, keep_prob_v):
    x, y_, keep_prob = placeholders
    return {x: data['features'], y_: data['labels'], keep_prob:keep_prob_v}

#to random eval a batch size in sequence
#for big dataset evaluate a bigger batch size
def do_batch_eval(sess, data_set, batch_size, accuracy, placeholders,
                  merged, test_writer, global_step, spannum=20,if_summary=True):
    indexs = get_random_seq_indexs(data_set)
    last_index = 0
    _, data, _ = sequence_get_data(data_set, indexs, last_index, batch_size, spannum)

    feed_dict = get_feed_dict(placeholders, data, 1.0)

    if if_summary == True:
        summary, result = sess.run([merged, accuracy], feed_dict=feed_dict)
        test_writer.add_summary(summary, global_step)
    else:
        result = sess.run(accuracy, feed_dict=feed_dict)

    return result

#to eval whole dataset
#not directly feed whole data
#feed batch size and get correct numbers
#at last calculate the whole dataset accuracy
def do_eval(sess, data_set, correct_num, placeholders, spannum=20):
    #fix batch size in 100
    batch_size = 100
    #total epoch loop num
    num_epoch = len(data_set) / batch_size
    #after the last loop, how many data rest
    rest_data_size = len(data_set) % batch_size

    #get the random index of dataset
    indexs = get_random_seq_indexs(data_set)
    last_index = 0
    count = 0
    for step in xrange(num_epoch):
        #will not out of dataset, not need care about it
        last_index, data, _ = sequence_get_data(data_set, indexs, last_index, batch_size, spannum)
        feed_dict = get_feed_dict(placeholders, data, 1.0)
        num = sess.run(correct_num, feed_dict=feed_dict)
        count += num

    if rest_data_size != 0:
        #the rest data
        last_index, data, _ = sequence_get_data(data_set, indexs, last_index, rest_data_size, spannum)
        feed_dict = get_feed_dict(placeholders, data, 1.0)
        num = sess.run(correct_num, feed_dict=feed_dict)
        count += num

    return count / data_set.shape[0]

#store our model
def store_model(last_accuracy, best_accuracy, best_path, dir_path, saver, sess, step):
    if last_accuracy > best_accuracy or last_accuracy == 1.0:
        best_accuracy = last_accuracy
        path = "modules/%s/%d/%.2f/model.ckpt" % (dir_path, step, last_accuracy)
        best_path = path
        del_and_create_dir(path)
        save_path = saver.save(sess, path)
        print("Model saved in file: %s" % save_path)
    return best_path, best_accuracy


#total train function
def train(y, y_one_hot, max_step, datasets, batch_size, sess, keep_prob_v, loss, accuracy,train_op, placeholders,
          lr_rate,
          lr_decay, lr_decay_epoch, correct_num, dir_path, merged, situation_now, loop, spannum=20, earlyStop=True):
    #get datasets
    train_dataset, validation_dataset, test_dataset = datasets

    #define store path
    train_path = 'tmp/train'
    test_path = 'tmp/test'
    del_and_create_dir(train_path)
    del_and_create_dir(test_path)

    #create writer for tensorboard
    train_writer = tf.summary.FileWriter(train_path, sess.graph)
    test_writer = tf.summary.FileWriter(test_path)

    #init all variables
    sess.run(tf.global_variables_initializer())

    #add saver to save model
    saver = tf.train.Saver()

    #best_accuracy means we will store model
    #which accuracy higher than 0.8 or equal 1.0
    last_accuracy = 0.0
    best_accuracy = 0.8
    best_path = ''

    #get a dataset indexs
    indexs = get_random_seq_indexs(train_dataset)

    #if out of dataset we reshuffle the dataset(reget the indexs)
    out_of_dataset = False
    last_index = 0
    train_acc = 0
    #to figure if need break
    break_result = 0

    log = ''
    # Train
    for step in xrange(max_step):
        before_time = time.time()

        if out_of_dataset == True:
            indexs = get_random_seq_indexs(train_dataset)
            last_index = 0
            out_of_dataset = False

        #get batch data
        last_index, data, out_of_dataset = sequence_get_data(train_dataset, indexs, last_index, batch_size, spannum)
        feed_dict = get_feed_dict(placeholders, data, keep_prob_v)

        #write summary every 40 steps and last step
        if step % 40 == 0 or step == max_step - 1:
            summary, _, loss_v = sess.run([merged, train_op, loss],feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
        else:
            _, loss_v = sess.run([train_op, loss],feed_dict=feed_dict)

        #write loss and time every 100 steps
        if step % 100 == 0:
            print '-----------loss in step %d is %f----------' % (step, loss_v)
            print sess.run(y[0], feed_dict=feed_dict)
            print sess.run(y_one_hot[0], feed_dict=feed_dict)


        #do evaluation every 500 steps and last step
        if step % 500 == 0 or step == max_step - 1:

            last_time = time.time()
            span_time = last_time - before_time
            print ('500 steps is %f second' % (span_time * 500))
            print ('rest time is %f minutes' % (span_time * (max_step - step) * loop / 60))

            result = do_batch_eval(
                sess, train_dataset, 1000, accuracy, placeholders,
                merged, test_writer, step, spannum, True)
            print '----------train acc in step %d is %f-------------' % (step, result)
            log += '\ntr a s %d %.4f' % (step, result)
            result = do_batch_eval(
                sess, validation_dataset, 1000, accuracy, placeholders,
                merged, test_writer, step, spannum, True)

            print '----------valid acc in step %d is %f-------------' % (step, result)

            #add log if good result then add whole log into file
            log += '\nva a s %d %.4f' % (step, result)

            #check the result, if better than best_accuracy then store model
            if result > last_accuracy:
                last_accuracy = result
                break_result = result
                best_path, best_accuracy = store_model(last_accuracy, best_accuracy, best_path, dir_path, saver, sess, step)

        #if 4000 steps not higher than 0.44
        #then we think this model is failed
        #end train
        if earlyStop == True:
            if (step % 6000 == 0 and step > 0):
                if break_result < 0.44:
                    train_writer.close()
                    test_writer.close()

                    return break_result

        #to decay learning rate
        if (step % lr_decay_epoch == 0 and step > 0):
            lr_rate *= lr_decay

    #use the best model to do the test evaluation
    #restore the best model
    if best_path != '':
        saver.restore(sess, best_path)
        print "Model restored."

    #do the last test evaluation
    result = do_eval(sess, validation_dataset, correct_num, placeholders, spannum)
    print '-----------last accuracy is %f------------' % (result)
    log += '\nte a %.4f\n' % (result)

    #store the good result model
    if result > 0.98:

        write_file(result, dir_path, situation_now+log)

        train_log_path = 'modules/%s/logs/train' % dir_path
        test_log_path = 'modules/%s/logs/test' % dir_path
        del_and_create_dir(train_log_path)
        del_and_create_dir(test_log_path)
        copyFiles(train_path, train_log_path)
        copyFiles(test_path, test_log_path)

    #store the bad model for check later
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

    return result
import pandas as pd
import numpy as np
from basic_model import *
from file_system_model import *
import otto_resnet_model as model
#the parameter need fill
#######################################################
#from network_model_example import *
epochs = 3
data_size = 61000
file_size = 61000
test_size = 878

batch_size = 100
#hypers
reg = 0.000067
lr_rate = 0.002
lr_decay = 0.99
keep_prob_v = 1.0
log_dir = 'logs/'
module_dir = 'modules/'
epoch = 0
loop = 0
best_model_number = 10
best_model_acc_dic = None
best_model_dir_dic = None
train_filename = 'data/norm/train_set.csv'
test_filename = 'data/norm/test_set.csv'

########################################################


#######################################################################
################      HELPER    #######################################
#######################################################################

def sequence_get_data(dataset, indexs, last_index, batch_size):
    next_index = last_index + batch_size

    if next_index > dataset.shape[0]:

        next_index -= dataset.shape[0]
        last_part = np.arange(last_index, indexs.shape[0])
        before_part = np.arange(next_index)
        span_index = indexs[np.concatenate((last_part, before_part))]  # link two parts together
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]
        out_of_dataset = False

    data = dataset[span_index]
    return (next_index, data, out_of_dataset)


def get_random_seq_indexs(data_set):
    data_size = data_set.shape[0]
    #index = tf.random_shuffle(tf.range(0, data_size))#maybe can do it on tensorflow later
    indexs = np.arange(data_size)
    np.random.shuffle(indexs)
    return indexs

def print_and_log(word, log):
    print word
    log += word + '\n'
    return log

################################################################################

# change the dir and log dir nameodule_dir = module_dir + model_name + '/'
log_dir = log_dir + 'otto_resnet' + '/'

create_dir(log_dir)
create_dir(module_dir)

loops = data_size // batch_size
test_loops = test_size // batch_size
test_rest = test_size % batch_size
test_loop = 0

log = ''

#hypers
lr_decay = 0.99
regs = random_uniform_array(7, -5, -3)
lr_rates = random_uniform_array(7, -5, -3)
#keeps = random_uniform_array(4, -0.3, 0)

log = ''

log_dir = 'logs/otto_resnet/'
del_and_create_dir(log_dir)

count_total = len(regs) * len(lr_rates)
count = 0

for reg in regs:
    for lr_rate in lr_rates:
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # log should update
                log = ''

                # reset the log dir
                log_dir = 'logs/otto_resnet/'

                # reset epoch and loop
                epoch = 0
                loop = 0

                # show hyper info
                word = '\nhyper\n'
                word += 'reg is %f\n' % reg
                word += 'lr_rate is %f\n' % lr_rate
                word += 'keep_prob_v is %f\n' % keep_prob_v
                log = print_and_log(word, log)

                # inputs
                input_x = tf.placeholder(tf.float32, [None, 96, 96, 93], name='input_x')
                input_y = tf.placeholder(tf.float32, [None, 9], name='input_y')
                train_phase = tf.placeholder(tf.bool, name='train_phase')
                keep_prob = tf.placeholder(tf.float32, name='keep_prob')

                # logits
                y_pred, parameters = model.inference(input_x, train_phase, keep_prob)

                # loss
                loss_value = loss(input_y, y_pred, reg, parameters)

                # train
                train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_value)

                # predict
                correct_num, accuracy = corr_num_acc(input_y, y_pred)

                sess.run(tf.global_variables_initializer())

                # read file
                train_data = pd.read_csv(train_filename, header=None).values
                test_data = pd.read_csv(test_filename, header=None).values
                test_x, test_y = model.reshape_dataset(test_data)
                while epoch < epochs:

                    print
                    word = 'total loop is %d, total epoch is %d' % (loops, epochs)
                    log = print_and_log(word, log)
                    word = 'here is the %d epoch' % epoch
                    log = print_and_log(word, log)
                    print

                    indexs = get_random_seq_indexs(train_data)
                    last_index = 0
                    out_of_dataset = False

                    while loop < loops:
                        before_time = time.time()

                        # should not happen
                        if out_of_dataset == True:
                            print "out of dataset"
                            indexs = get_random_seq_indexs(train_data)
                            last_index = 0
                            out_of_dataset = False

                        last_index, data, out_of_dataset = sequence_get_data(train_data, indexs, last_index,
                                                                             batch_size)

                        data_x, data_y = model.reshape_dataset(data)

                        feed_dict = {input_x: data_x, input_y: data_y, train_phase: True, keep_prob: keep_prob_v}
                        _, loss_v, acc = sess.run([train_step, loss_value, accuracy], feed_dict=feed_dict)

                        word = 'loss in loop %d is %.4f, acc is %.4f' % (loop, loss_v, acc)
                        log = print_and_log(word, log)

                        loop += 1

                        # show time
                        if loop % 50 == 0:
                            print
                            word = 'total loop is %d, total epoch is %d' % (loops, epochs)
                            log = print_and_log(word, log)
                            word = 'here is the %d epoch' % epoch
                            log = print_and_log(word, log)
                            print

                            last_time = time.time()
                            span_time = last_time - before_time
                            rest_loop = loops - loop
                            rest_epoch = epochs - epoch
                            last_loop_num = 10

                            # show the last loop time
                            word = 'last %d loop use %f minutes' % (last_loop_num, span_time * last_loop_num / 60)
                            log = print_and_log(word, log)

                            # show the rest loop time
                            word = 'rest loop need %.3f minutes' % (span_time * rest_loop / 60)
                            log = print_and_log(word, log)

                            # show the rest epoch time
                            word = 'rest epoch need %.3f hours' % (
                                span_time * rest_loop / 3600 + span_time * loops * rest_epoch / 3600)
                            log = print_and_log(word, log)

                            # show the rest total time
                            rest_count = count_total - count
                            word = 'rest total time need %.3f hours' % (
                                span_time * rest_loop / 3600 + span_time * loops * rest_epoch / 3600 + span_time *
                                loops * epochs * rest_count / 3600)
                            log = print_and_log(word, log)
                            print



                    # do the test
                    indexs = get_random_seq_indexs(test_data)
                    out_of_dataset = False
                    last_index = 0
                    total_acc_num = 0.0
                    total_loss = 0.0

                    while test_loop < test_loops:
                        print '%d ' % test_loop,

                        last_index, data, out_of_dataset = sequence_get_data(test_data, indexs, last_index,
                                                                             batch_size)

                        data_x, data_y = model.reshape_dataset(data)

                        feed_dict = {input_x: data_x, input_y: data_y, train_phase: False, keep_prob: keep_prob_v}
                        corr_num, test_loss = sess.run([correct_num, loss_value], feed_dict=feed_dict)
                        total_acc_num += corr_num
                        total_loss += test_loss

                        test_loop += 1

                    if test_rest != 0:
                        span_index = indexs[last_index:]
                        data = test_data[span_index]
                        data_x, data_y = model.reshape_dataset(data)
                        feed_dict = {input_x: data_x, input_y: data_y, train_phase: False, keep_prob: keep_prob_v}
                        corr_num, test_loss = sess.run([correct_num, loss_value], feed_dict=feed_dict)
                        total_acc_num += corr_num
                        total_loss += test_loss

                    acc = total_acc_num / test_size
                    test_loss = total_loss / test_size
                    word = 'test acc in epoch %d is %.4f, loss is %.4f' % (epoch, acc, test_loss)
                    log = print_and_log(word, log)

                    # to create the count dir
                    if epoch == 0:
                        log_dir += '%.4f_count%d/' % (acc, count)
                        del_and_create_dir(log_dir)

                        filename = log_dir + 'hypers'
                        hyper_info = '\nhyper\n'
                        hyper_info += 'reg is %f\n' % reg
                        hyper_info += 'lr_rate is %f\n' % lr_rate
                        hyper_info += 'keep_prob_v is %f\n' % keep_prob_v
                        f = file(filename, 'w+')
                        f.write(hyper_info)
                        f.close()

                    filename = log_dir + '%.4f_ls%.4f_epoch%d' % (acc, test_loss, epoch)
                    f = file(filename, 'w+')
                    f.write(log)
                    f.close()

                    # reset loop
                    loop = 0
                    test_loop = 0
                    # each epoch decay the lr_rate
                    lr_rate *= lr_decay

                    epoch += 1

        count += 1

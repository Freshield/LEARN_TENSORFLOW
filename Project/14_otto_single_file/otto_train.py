from basic_model import *
from file_system_model import *
import otto_resnet_model as model
import pandas as pd
import numpy as np
#the parameter need fill
#######################################################
#from network_model_example import *
dir = 'data/norm/train1000/'
epochs = 200
data_size = 61000
file_size = 61000

batch_size = 100
#hypers
reg = 0.000067
lr_rate = 0.002
lr_decay = 0.99
keep_prob_v = 0.9569
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

################################################################################

# change the dir and log dir nameodule_dir = module_dir + model_name + '/'
log_dir = log_dir + 'otto_resnet' + '/'

loops = data_size // batch_size

with tf.Graph().as_default():
    with tf.Session() as sess:
        # inputs
        input_x = tf.placeholder(tf.float32, [None, 96, 96, 1], name='input_x')
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
            before_time = time.time()


            indexs = get_random_seq_indexs(train_data)
            last_index = 0
            out_of_dataset = False

            while loop < loops:

                # should not happen
                if out_of_dataset == True:
                    print "out of dataset"
                    indexs = get_random_seq_indexs(train_data)
                    last_index = 0
                    out_of_dataset = False

                last_index, data, out_of_dataset = sequence_get_data(train_data, indexs, last_index, batch_size)

                data_x, data_y = model.reshape_dataset(data)

                feed_dict = {input_x: data_x, input_y: data_y, train_phase: True,
                             keep_prob: keep_prob_v}
                _, loss_v, acc = sess.run([train_step, loss_value, accuracy], feed_dict=feed_dict)

                print 'loss in loop %d is %.4f, acc is %.4f' % (loop, loss_v, acc)

                loop += 1

            #do the test


            feed_dict = {input_x: test_x, input_y: test_y, train_phase: True, keep_prob: keep_prob_v}
            acc = sess.run(accuracy, feed_dict=feed_dict)

            print 'test acc in epoch %d is %.4f' % acc

            # reset loop
            loop = 0
            # each epoch decay the lr_rate
            lr_rate *= lr_decay

            epoch += 1

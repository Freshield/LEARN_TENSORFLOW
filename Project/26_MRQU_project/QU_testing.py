from basic_model import *
from file_system_model import *
import QU_resnet_model as model
import pandas as pd
import numpy as np
#the parameter need fill
#######################################################
#from network_model_example import *
epochs = 200
data_size = 24000
file_size = 24000
test_size = 6000

batch_size = 100
#hypers
reg = 0.000043
lr_rate = 0.000034
lr_decay = 0.99
keep_prob_v = 1.0
log_dir = 'logs/'
module_dir = 'modules/'
epoch = 0
loop = 0
best_model_number = 10
best_model_acc_dic = None
best_model_dir_dic = None
train_filename = 'data/clean_raw_train_35.csv'
test_filename = 'data/clean_raw_test_35.csv'
ones_filename = 'data/train_label_one_set.csv'
ones_batch = 40
########################################################


#######################################################################
################      HELPER    #######################################
#######################################################################

def get_data_from_label_one_set(filename,batch_num):
    data = pd.read_csv(filename,header=None).values

    indexs = np.arange(data.shape[0])
    np.random.shuffle(indexs)

    ones_data = data[indexs[:batch_num]]
    return ones_data

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
log_dir = log_dir + 'QU_resnet' + '/'

create_dir(log_dir)
create_dir(module_dir)

loops = data_size // batch_size
test_loops = test_size // batch_size
test_rest = test_size % batch_size
test_loop = 0

log = ''
model_path = 'model/module.ckpt'

# read file
test_data = pd.read_csv(test_filename).values

with tf.Graph().as_default():
    with tf.Session() as sess:
        # inputs
        input_x = tf.placeholder(tf.float32, [None, 40, 40, 35], name='input_x')
        input_y = tf.placeholder(tf.float32, [None, 2], name='input_y')
        train_phase = tf.placeholder(tf.bool, name='train_phase')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # logits
        y_pred, parameters = model.inference(input_x, train_phase, keep_prob)

        y_type = tf.argmax(y_pred, axis=1)

        y_true = tf.argmax(input_y, axis=1)

        # loss
        loss_value = loss(input_y, y_pred, reg, parameters)

        # train
        train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_value)

        # predict
        correct_num, accuracy = corr_num_acc(input_y, y_pred)

        saver = tf.train.Saver()

        saver.restore(sess, model_path)

        print ''
        print 'Model was restored'

        indexs = np.arange(test_data.shape[0])
        out_of_dataset = False
        last_index = 0
        total_acc_num = 0.0
        total_loss = 0.0

        result_matrix = np.zeros((test_data.shape[0],2))
        result_index = 0

        y_pred_matrix = np.zeros((test_data.shape[0],2))
        y_label_matrix = np.zeros((test_data.shape[0],2))

        while test_loop < test_loops:
            print '%d ' % test_loop,

            last_index, data, out_of_dataset = sequence_get_data(test_data, indexs, last_index, batch_size)

            data_x, data_y = model.reshape_dataset(data)

            feed_dict = {input_x: data_x, input_y: data_y, train_phase: False,
                         keep_prob: keep_prob_v}

            corr_num, test_loss, y_type_v, y_true_v, y_pred_v = sess.run([correct_num, loss_value, y_type, y_true, y_pred],
                                                               feed_dict=feed_dict)
            result_matrix[result_index:result_index+batch_size,0] = y_type_v
            result_matrix[result_index:result_index+batch_size,1] = y_true_v

            y_pred_matrix[result_index:result_index+batch_size,:] = y_pred_v
            y_label_matrix[result_index:result_index+batch_size,:] = data_y

            result_index += batch_size

            total_acc_num += corr_num
            total_loss += test_loss

            test_loop += 1

        if test_rest != 0:
            span_index = indexs[last_index:]
            data = test_data[span_index]
            data_x, data_y = model.reshape_dataset(data)
            feed_dict = {input_x: data_x, input_y: data_y, train_phase: True,
                         keep_prob: keep_prob_v}
            corr_num, test_loss, y_type_v, y_true_v, y_pred_v = sess.run([correct_num, loss_value, y_type, y_true, y_pred],
                                                               feed_dict=feed_dict)
            result_matrix[result_index:result_index+batch_size,0] = y_type_v
            result_matrix[result_index:result_index+batch_size,1] = y_true_v

            y_pred_matrix[result_index:result_index+batch_size,:] = y_pred_v
            y_label_matrix[result_index:result_index+batch_size,:] = data_y

            result_index += batch_size

            total_acc_num += corr_num
            total_loss += test_loss

        acc = total_acc_num / test_size
        test_loss = total_loss / test_size

        word = 'test acc in epoch %d is %.4f, loss is %.4f' % (epoch, acc, test_loss)
        log = print_and_log(word, log)

        header = 'predict_value,true_value'
        np.savetxt('result_value.csv', result_matrix, delimiter=',', header=header, comments='')
        np.savetxt('y_pred_value.csv', y_pred_matrix, delimiter=',')
        np.savetxt('y_label_value.csv', y_label_matrix, delimiter=',')
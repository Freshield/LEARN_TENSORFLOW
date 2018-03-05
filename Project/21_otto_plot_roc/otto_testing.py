from basic_model import *
from file_system_model import *
import otto_resnet_simple_model_2 as model
import pandas as pd
import numpy as np
#the parameter need fill
#######################################################
#from network_model_example import *
test_size = 878

batch_size = 100
#hypers
reg = 0.12789
lr_rate = 0.000094
lr_decay = 0.99
keep_prob_v = 1.0
model_path = '/home/freshield/ciena_test/20_otto_test_dataset/modules/model/module.ckpt'
test_filename = '/media/freshield/CORSAIR/LEARN_TENSORFLOW/Project/14_otto_single_file/data/norm/test_set.csv'
########################################################


#######################################################################
################      HELPER    #######################################
#######################################################################

def sequence_get_data(dataset, indexs, last_index, batch_size):
    next_index = last_index + batch_size

    if next_index > dataset.shape[0]:
        span_index = indexs[last_index:]
        out_of_dataset = True
    else:
        span_index = indexs[last_index:next_index]
        out_of_dataset = False

    data = dataset[span_index]
    return (next_index, data, out_of_dataset)


################################################################################

# change the dir and log dir nameodule_dir = module_dir + model_name + '/'

test_loops = test_size // batch_size
test_rest = test_size % batch_size
test_loop = 0

result_dataset = np.zeros((test_size,9))
label_dataset = np.zeros((test_size,9))

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

        saver = tf.train.Saver()

        saver.restore(sess, model_path)

        print ''
        print 'Model was restored'

        # read file
        test_data = pd.read_csv(test_filename, header=None).values

        # do the test
        res_last_index = 0
        res_next_index = 0

        total_acc_num = 0
        total_loss = 0

        indexs = np.arange(test_size)

        while test_loop < test_loops:
            print '%d ' % test_loop,

            res_next_index = res_last_index + batch_size

            data = test_data[res_last_index:res_next_index,:]

            data_x, data_y = model.reshape_dataset(data)

            feed_dict = {input_x: data_x, input_y: data_y, train_phase: True,
                         keep_prob: keep_prob_v}
            y_pred_v, corr_num, test_loss = sess.run([y_pred,correct_num, loss_value], feed_dict=feed_dict)
            total_acc_num += corr_num
            total_loss += test_loss

            result_dataset[res_last_index:res_next_index,:] = y_pred_v
            label_dataset[res_last_index:res_next_index, :] = data_y

            res_last_index = res_next_index

            test_loop += 1

        if test_rest != 0:
            data = test_data[res_last_index:, :]

            data_x, data_y = model.reshape_dataset(data)
            feed_dict = {input_x: data_x, input_y: data_y, train_phase: True,
                         keep_prob: keep_prob_v}
            y_pred_v,corr_num, test_loss = sess.run([y_pred,correct_num, loss_value], feed_dict=feed_dict)
            total_acc_num += corr_num
            total_loss += test_loss

            result_dataset[res_last_index:, :] = y_pred_v
            label_dataset[res_last_index:, :] = data_y

        acc = total_acc_num / test_size
        test_loss = total_loss / test_size
        print 'acc is %.4f' % acc
        print 'loss is %.4f' % test_loss

        np.savetxt('result_T.csv', result_dataset, delimiter=',')
        np.savetxt('label_T.csv', label_dataset, delimiter=',')





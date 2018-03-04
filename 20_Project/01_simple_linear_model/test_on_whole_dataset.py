import pandas as pd
import simple_linear as sl
import tensorflow as tf
import file_filter as ff

################################################
#                                              #
#           use best model json file           #
#      create model to train on whole dataset  #
#                                              #
################################################



filename = 'ciena_test.csv'

#just show the parameters
#we will use the value in json file
batch_size = 100
lr_rate = 0.01
max_step = 10000
reg = 0.02
lr_decay = 0.97
lr_decay_epoch = 1500
keep_prob_v = 1.0
stddev = 0.8
use_L2 = False
hidden1_size = 200
hidden2_size = 100
hidden3_size = 30

#get the dataset
dataset = pd.read_csv(filename, header=None)
train_dataset, validation_dataset, test_dataset = sl.split_dataset(dataset, radio=0.1)

datasets = (train_dataset, validation_dataset, test_dataset)

#get the dictionary list from json files
best_para_list = ff.json_to_dic_list('json')

#loop value for calculate time
loop = len(best_para_list)

#for loop to train each parameters
for paras in best_para_list:
    reg = paras['reg']
    lr_rate = paras['lr_rate']
    stddev = paras['stddev']
    hidden1_size = int(paras['hidden1'])
    hidden2_size = int(paras['hidden2'])
    hidden3_size = int(paras['hidden3'])
    keep_prob_v = paras['keep_prob']
    use_L2 = paras['L2']
    print '-------------------now changed-----------------'
    print 'reg is', reg
    print 'lr_rate is', lr_rate
    print 'stddev is', stddev
    print 'hidden1 size is', hidden1_size
    print 'hidden2 size is', hidden2_size
    print 'hidden3 size is', hidden3_size
    print 'keep prob is', keep_prob_v
    print 'use L2 is', use_L2
    print 'rest loop is', loop
    print '------------------------------------------------'

    situation_now = '\n-------------------now changed-----------------\n' \
                    'reg is %f\nlr_rate is %f\nstddev is %f\n' \
                    'hidden1 size is %d\nhidden2 size is %d\n' \
                    'hidden3 size is %d\nuse L2 is %s\nkeep_prob is %.2f' \
                    '------------------------------------------------' % (
                        reg, lr_rate, stddev, hidden1_size,
                        hidden2_size, hidden3_size, use_L2, keep_prob_v)
    dir_path = "r%.2flr%.2fs%.2fkp%.2fh%d%d%du%s" % (
        reg, lr_rate, stddev, keep_prob_v,
        hidden1_size, hidden2_size, hidden3_size, use_L2)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # create graph
            x, y_, y_one_hot = sl.get_inputs()
            y, l2_loss, keep_prob = sl.get_logits(
                x, hidden1_size, hidden2_size, hidden3_size,
                3, stddev, stddev
            )
            loss = sl.get_loss(y, y_one_hot, l2_loss, reg, True, use_L2)
            train_op = sl.get_train_op(loss, lr_rate)
            correct_num = sl.get_correct_num(y, y_one_hot)
            accuracy = sl.get_accuracy(y, y_one_hot)

            #store all of the summary
            merged = tf.summary.merge_all()

            placeholders = (x, y_, keep_prob)
            # train
            last_result = sl.train(max_step, datasets, batch_size, sess, keep_prob_v,
                                   loss, accuracy, train_op, placeholders, lr_rate,
                                   lr_decay, lr_decay_epoch, correct_num, dir_path,
                                   merged, situation_now, loop)

            #if not have good result and was stopped
            #give this model one more chance
            if last_result < 0.6:
                last_result = sl.train(max_step, datasets, batch_size, sess, keep_prob_v,
                                       loss, accuracy, train_op, placeholders, lr_rate,
                                       lr_decay, lr_decay_epoch, correct_num, dir_path,
                                       merged, situation_now, loop)

            #if not have good result and was stopped
            #give this model one more chance
            if last_result < 0.6:
                last_result = sl.train(max_step, datasets, batch_size, sess, keep_prob_v,
                                       loss, accuracy, train_op, placeholders, lr_rate,
                                       lr_decay, lr_decay_epoch, correct_num, dir_path,
                                       merged, situation_now, loop)

            loop -= 1
import time
from basic_model import *
from data_process import *
from file_system_model import *

#the parameter need fill
#######################################################
from network_example import *
SPAN=[10]
dir = '/media/freshield/SOFTWARE3/Ciena/raw/norm/'
epochs = 100
data_size = 600000
file_size = 1000
#how many loops do an evaluation
loop_eval_num = 50
batch_size = 100
train_file_size = 800
valid_file_size = 100
test_file_size = 100
#hypers
reg = 0.000067
lr_rate = 0.002
lr_decay = 0.99
keep_prob_v = 0.9569
log_dir = 'logs/Resnet_link_test/'
########################################################

max_step = train_file_size // batch_size

loops = data_size // file_size

log = ''

del_and_create_dir(log_dir)

with tf.Graph().as_default():
    with tf.Session() as sess:
        #inputs
        input_x = tf.placeholder(tf.float32, [None, 32, 104, 2], name='input_x')
        para_pl = tf.placeholder(tf.float32, [None, 41], name='para_pl')
        input_y = tf.placeholder(tf.float32, [None, 3], name='input_y')
        train_phase = tf.placeholder(tf.bool, name='train_phase')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        #logits
        y_pred, parameters = inference(input_x, para_pl, train_phase, keep_prob)

        #loss
        loss_value = loss(input_y, y_pred, reg, parameters)

        #train
        train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_value)

        #predict
        correct_num, accuracy = corr_num_acc(input_y, y_pred)

        #placeholders
        placeholders = (input_x, para_pl, input_y, train_phase, keep_prob)
        train_pl = input_x, para_pl, input_y, train_phase, keep_prob, train_step, loss_value, accuracy

        sess.run(tf.global_variables_initializer())

        for epoch in xrange(epochs):

            #show the epoch num
            words_log_print_epoch(epoch, epochs, log)

            loop_indexs = get_file_random_seq_indexs(loops)

            #caution loop is not in sequence
            for loop in xrange(loops):
                before_time = time.time()

                train_file = "train_set_%d.csv"  % loop_indexs[loop]

                loop_loss_v, loop_acc = do_train_file(sess, train_pl, dir, train_file, SPAN, max_step, batch_size, keep_prob_v)

                words_log_print_loop(loop, loops, loop_loss_v, loop_acc, log)

                #each 50 loop, do evaluation
                if (loop != 0 and loop % loop_eval_num == 0) or loop == loops - 1:
                    #show the time
                    time_show(before_time, loop_eval_num, loop, loops, epoch, epochs, log)
                    #store the parameter first
                    eval_parameters = (loop, loop_indexs, SPAN, sess, batch_size, correct_num, placeholders, log)
                    #here only evaluate last 10 files
                    evaluate_last_x_files(10, eval_parameters)

            #each epoch decay the lr_rate
            lr_rate *= lr_decay

            #store the parameter first
            test_parameter = loops, epoch, SPAN, sess, batch_size, correct_num, placeholders, log
            test_acc = evaluate_test(test_parameter)

            store_log(log_dir, test_acc, epoch, log)
import time
#from basic_model import *
from data_process_model import *
from file_system_model import *

#the parameter need fill
#######################################################
#from network_model_example import *
SPAN=[10]
dir = '/home/freshield/Ciena_data/dataset_10k/model/'
epochs = 20
data_size = 10000
file_size = 1000
#how many loops do an evaluation
loop_eval_num = 5
#how many file do the valid
eval_last_num = 10
batch_size = 100
train_file_size = 800
valid_file_size = 100
test_file_size = 100
#hypers
reg = 0.000067
lr_rate = 0.002
lr_decay = 0.99
keep_prob_v = 0.9569
log_dir = 'logs/Link_CNN/'
module_dir = 'modules/Link_CNN/'
########################################################

para_whole_dataset_dic = {
    'SPAN' : SPAN,
    'dir' : dir,
    'epochs' : epochs,
    'data_size' : data_size,
    'file_size' : file_size,
    'loop_eval_num' : loop_eval_num,
    'batch_size' : batch_size,
    'train_file_size' : train_file_size,
    'valid_file_size' : valid_file_size,
    'test_file_size' : test_file_size,
    'reg' : reg,
    'lr_rate' : lr_rate,
    'lr_decay' : lr_decay,
    'keep_prob_v' : keep_prob_v,
    'log_dir' : log_dir,
    'module_dir' : module_dir,
    'eval_last_num' : eval_last_num
}

#retrive the para from dic
#ver 1.0
def get_whole_dataset_para(para_dic):
    SPAN = para_dic['SPAN']
    dir = para_dic['dir']
    epochs = para_dic['epochs']
    data_size = para_dic['data_size']
    file_size = para_dic['file_size']
    loop_eval_num = para_dic['loop_eval_num']
    batch_size = para_dic['batch_size']
    train_file_size = para_dic['train_file_size']
    valid_file_size = para_dic['valid_file_size']
    test_file_size = para_dic['test_file_size']
    reg = para_dic['reg']
    lr_rate = para_dic['lr_rate']
    keep_prob_v = para_dic['keep_prob_v']
    log_dir = para_dic['log_dir']
    module_dir = para_dic['module_dir']
    eval_last_num = para_dic['eval_last_num']
    return SPAN,dir,epochs,data_size,file_size,loop_eval_num,batch_size,train_file_size,valid_file_size,test_file_size,reg,lr_rate,keep_prob_v,log_dir,module_dir,eval_last_num

#train the model
#ver 1.0
def train_whole_dataset_begin(para_dic, model_name):

    #choose model
    if model_name == 'resnet_link':
        from Resnet_link_model import *
    elif model_name == 'link_cnn':
        from Link_CNN_model import *
    else:
        print "Error model name"
        return 'error'

    #get all para first
    SPAN, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size, valid_file_size, test_file_size, reg, lr_rate, keep_prob_v, log_dir, module_dir, eval_last_num = get_whole_dataset_para(para_dic)

    max_step = train_file_size // batch_size
    loops = data_size // file_size
    log = ''
    del_and_create_dir(log_dir)
    del_and_create_dir(module_dir)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # inputs
            input_x = tf.placeholder(tf.float32, [None, 32, 104, 2], name='input_x')
            para_pl = tf.placeholder(tf.float32, [None, 41], name='para_pl')
            input_y = tf.placeholder(tf.float32, [None, 3], name='input_y')
            train_phase = tf.placeholder(tf.bool, name='train_phase')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # logits
            y_pred, parameters = inference(input_x, para_pl, train_phase, keep_prob)

            # loss
            loss_value = loss(input_y, y_pred, reg, parameters)

            # train
            train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_value)

            # predict
            correct_num, accuracy = corr_num_acc(input_y, y_pred)

            # placeholders
            placeholders = (input_x, para_pl, input_y, train_phase, keep_prob)
            train_pl = input_x, para_pl, input_y, train_phase, keep_prob, train_step, loss_value, accuracy

            sess.run(tf.global_variables_initializer())

            for epoch in xrange(epochs):

                # show the epoch num
                words_log_print_epoch(epoch, epochs, log)

                loop_indexs = get_file_random_seq_indexs(loops)

                # caution loop is not in sequence
                for loop in xrange(loops):
                    before_time = time.time()

                    train_file = "train_set_%d.csv" % loop_indexs[loop]

                    loop_loss_v, loop_acc = do_train_file(sess, train_pl, dir, train_file, SPAN, max_step, batch_size,
                                                          keep_prob_v)

                    words_log_print_loop(loop, loops, loop_loss_v, loop_acc, log)

                    # each loop_eval_num, do evaluation
                    if (loop != 0 and loop % loop_eval_num == 0) or loop == loops - 1:
                        # show the time
                        time_show(before_time, loop_eval_num, loop, loops, epoch, epochs, log)
                        # store the parameter first
                        eval_parameters = (loop, loop_indexs, SPAN, sess, batch_size, correct_num, placeholders, log)
                        # here only evaluate last eval_last_num files
                        evaluate_last_x_files(eval_last_num, eval_parameters, dir)

                # each epoch decay the lr_rate
                lr_rate *= lr_decay

                # store the parameter first
                test_parameter = loops, epoch, SPAN, sess, batch_size, correct_num, placeholders, log, dir
                # do the test evaluate
                test_acc = evaluate_test(test_parameter)

                # store log file every epoch
                store_log(log_dir, test_acc, epoch, log)
                # store module every epoch
                store_module(module_dir, test_acc, epoch, sess, log)

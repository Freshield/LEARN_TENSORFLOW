from file_system_model import *
from basic_model import *
import flow_model as fm

#the parameter need fill
#######################################################
#from network_model_example import *
SPAN=[5]
dir = '/home/freshield/Ciena_data/dataset_10k/model/'
epochs = 200
data_size = 1000000
file_size = 1000
#how many loops do an evaluation
loop_eval_num = 50
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
keep_prob_v = 1.0
log_dir = 'logs/'
module_dir = 'modules/'
epoch = 0
loop = 0
best_model_number = 10
best_model_acc_dic = None
best_model_dir_dic = None
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
    'eval_last_num' : eval_last_num,
    'epoch' : epoch,
    'loop' : loop,
    'best_model_number' : best_model_number,
    'best_model_acc_dic' : best_model_acc_dic,
    'best_model_dir_dic' : best_model_dir_dic
}

#train the model
#ver 1.0
def train_whole_dataset_begin(para_dic, model_name):

    #choose model
    if model_name == 'resnet_link':
        import Resnet_link_model as rl
        model = rl
    elif model_name == 'link_cnn':
        import Link_CNN_model as lc
        model = lc
    else:
        print "Error model name"
        return 'error'


    dpm.model = model

    #get all para first
    [SPAN, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size, valid_file_size, test_file_size, reg, lr_rate, lr_decay, keep_prob_v, log_dir, module_dir, eval_last_num, epoch, loop, best_model_number, best_model_acc_dic, best_model_dir_dic] = get_para_from_dic(para_dic)

    #change the dir and log dir nameodule_dir = module_dir + model_name + '/'
    log_dir = log_dir + model_name + '/'

    #create the acc dic and dir dic
    best_model_acc_dic = np.arange(0.0,-best_model_number,-1.0).tolist()
    best_model_dir_dic = []
    for i in range(best_model_number):
        best_model_dir_dic.append('%s'%best_model_acc_dic[i])

    max_step = train_file_size // batch_size
    loops = data_size // file_size
    log = Log()
    create_dir(log_dir)
    create_dir(module_dir)

    words = 'Begin to train\n'
    words += time.strftime('%Y-%m-%d %H:%M:%S\n')
    words_log_print(words)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # inputs
            input_x = tf.placeholder(tf.float32, [None, 304, 48, 2], name='input_x')
            para_pl = tf.placeholder(tf.float32, [None, 21], name='para_pl')
            input_y = tf.placeholder(tf.float32, [None, 6], name='input_y')
            train_phase = tf.placeholder(tf.bool, name='train_phase')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # logits
            y_pred, parameters = model.inference(input_x, para_pl, train_phase, keep_prob)

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

            while epoch < epochs:

                words = time.strftime('%Y-%m-%d %H:%M:%S\n')
                words_log_print(words, log)

                # show the epoch num
                words_log_print_epoch(epoch, epochs, log)

                loop_indexs = dpm.get_file_random_seq_indexs(loops)

                # caution loop is not in sequence

                while loop < loops:
                    before_time = time.time()

                    train_file = "Raw_data_%d_train.csv" % loop_indexs[loop]

                    loop_loss_v, loop_acc = do_train_file(sess, train_pl, dir, train_file, SPAN, max_step, batch_size,keep_prob_v)

                    words_log_print_loop(loop, loops, loop_loss_v, loop_acc, log)

                    loop += 1

                    # each loop_eval_num, do evaluation
                    if loop % loop_eval_num == 0 or loop == loops:

                        words = time.strftime('%Y-%m-%d %H:%M:%S\n')
                        words_log_print(words, log)

                        # show the time
                        time_show(before_time, loop_eval_num, loop, loops, epoch, epochs, log)
                        # store the parameter first
                        eval_parameters = (loop, loop_indexs, SPAN, sess, batch_size, correct_num, placeholders, log)
                        # here only evaluate last eval_last_num files
                        evaluate_last_x_files(eval_last_num, eval_parameters, dir)

                        #ask for if want to interrupt
                        #press i to interrupt
                        temp_para = [SPAN, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size, valid_file_size, test_file_size, reg, lr_rate, lr_decay, keep_prob_v, log_dir, module_dir, eval_last_num, epoch, loop, best_model_number, best_model_acc_dic, best_model_dir_dic]

                        answer = fm.interrupt_flow(temp_para, sess, log, loop_indexs)
                        if answer == 'Done':
                            return 'Done'

                        [SPAN, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size,valid_file_size, test_file_size, reg, lr_rate, lr_decay, keep_prob_v, log_dir, module_dir,eval_last_num, epoch, loop, best_model_number, best_model_acc_dic, best_model_dir_dic] = temp_para




                #reset loop
                loop = 0
                # each epoch decay the lr_rate
                lr_rate *= lr_decay

                # store the parameter first
                test_parameter = loops, epoch, SPAN, sess, batch_size, correct_num, placeholders, log, dir
                # do the test evaluate
                test_acc = evaluate_test(test_parameter)

                temp_best_acc = np.array(best_model_acc_dic)
                #only store x best model
                if test_acc > temp_best_acc.min():
                    small_index = temp_best_acc.argmin()
                    temp_best_acc[small_index] = test_acc
                    module_path = module_dir + "%.4f_epoch%d/" % (test_acc, epoch)
                    #delete the latest module
                    del_dir(best_model_dir_dic[small_index])
                    best_model_dir_dic[small_index] = module_path
                    best_model_acc_dic = temp_best_acc.tolist()
                    # store module every epoch
                    store_module(module_dir, test_acc, epoch, sess, log, loop_indexs)

                # store log file every epoch
                store_log(log_dir, test_acc, epoch, log)

                epoch += 1
    return 'Done'
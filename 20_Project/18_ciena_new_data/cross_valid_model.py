from file_system_model import *
from basic_model import *
import flow_model as fm

#the parameter need fill
#######################################################
#from network_model_example import *
SPAN=[5]
dir = '/media/freshield/DATA_W/Ciena_new_data/10spans_norm/'
epochs = 1
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
best_model_number = 5
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


def cross_valid(para_dic):


    import Resnet_link_model as rl
    model = rl

    dpm.model = model
    # get all para first
    [SPAN, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size, valid_file_size,test_file_size,reg, lr_rate, lr_decay, keep_prob_v, log_dir, module_dir, eval_last_num, epoch, loop, best_model_number,best_model_acc_dic, best_model_dir_dic] = get_para_from_dic(para_dic)

    # change the dir and log dir nameodule_dir = module_dir + model_name + '/'

    log_dir = 'logs/Restnet_link_cross_valid/'

    # create the acc dic and dir dic
    best_model_acc_dic = np.arange(0.0, -best_model_number, -1.0).tolist()
    best_model_dir_dic = []
    for i in range(best_model_number):
        best_model_dir_dic.append('%s' % best_model_acc_dic[i])

    max_step = train_file_size // batch_size
    loops = data_size // file_size


    # hypers
    regs = random_uniform_array(7, -5, -1)
    lr_rates = random_uniform_array(7, -7, -2)

    count_total = len(regs) * len(lr_rates)
    count = 0

    print 'Begin to cross valid'
    print time.strftime('%Y-%m-%d %H:%M:%S')

    for reg in regs:
        for lr_rate in lr_rates:
            log = Log()

            log_dir = 'logs/Restnet_link_cross_valid/r%.4f_l%.4f_count%d/' % (reg,lr_rate,count)
            module_dir = 'modules/Restnet_link_cross_valid/r%.4f_l%.4f_count%d/' % (reg, lr_rate,count)
            create_dir(log_dir)
            create_dir(module_dir)

            # show hyper info
            words = '\nhyper\n'
            words += 'reg is %f\n' % reg
            words += 'lr_rate is %f\n' % lr_rate
            words += 'keep_prob_v is %f\n' % keep_prob_v
            words_log_print(words, log)

            filename = log_dir + 'hypers'
            hyper_info = '\nhyper\n'
            hyper_info += 'reg is %f\n' % reg
            hyper_info += 'lr_rate is %f\n' % lr_rate
            hyper_info += 'keep_prob_v is %f\n' % keep_prob_v
            f = file(filename, 'w+')
            f.write(hyper_info)
            f.close()



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

                        words = time.strftime('%Y-%m-%d %H:%M:%S')
                        words_log_print(words, log)

                        # show the epoch num
                        words_log_print_epoch(epoch, epochs, log)

                        loop_indexs = dpm.get_file_random_seq_indexs(loops)

                        # caution loop is not in sequence

                        while loop < loops:
                            before_time = time.time()

                            train_file = "Raw_data_%d_train.csv" % loop_indexs[loop]

                            loop_loss_v, loop_acc = do_train_file(sess, train_pl, dir, train_file, SPAN, max_step,batch_size, keep_prob_v)

                            words_log_print_loop(loop, loops, loop_loss_v, loop_acc, log)

                            loop += 1

                            # each loop_eval_num, do evaluation
                            if loop % loop_eval_num == 0 or loop == loops:

                                words = time.strftime('%Y-%m-%d %H:%M:%S')
                                words_log_print(words, log)

                                # show the time
                                time_show(before_time, loop_eval_num, loop, loops, epoch, epochs, log, count, count_total)
                                # store the parameter first
                                eval_parameters = (loop, loop_indexs, SPAN, sess, batch_size, correct_num, placeholders, log)
                                # here only evaluate last eval_last_num files
                                evaluate_last_x_files(eval_last_num, eval_parameters, dir)

                                # ask for if want to interrupt
                                # press i to interrupt
                                temp_para = [SPAN, dir, epochs, data_size, file_size, loop_eval_num, batch_size,train_file_size, valid_file_size, test_file_size, reg, lr_rate, lr_decay,keep_prob_v, log_dir, module_dir, eval_last_num, epoch, loop,best_model_number, best_model_acc_dic, best_model_dir_dic]

                                answer = fm.interrupt_flow(temp_para, sess, log, loop_indexs)
                                if answer == 'Done':
                                    return 'Done'

                                [SPAN, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size,valid_file_size, test_file_size, reg, lr_rate, lr_decay, keep_prob_v, log_dir,module_dir, eval_last_num, epoch, loop, best_model_number, best_model_acc_dic,best_model_dir_dic] = temp_para


                        # reset loop
                        loop = 0
                        # each epoch decay the lr_rate
                        lr_rate *= lr_decay

                        # store the parameter first
                        test_parameter = loops, epoch, SPAN, sess, batch_size, correct_num, placeholders, log, dir
                        # do the test evaluate
                        test_acc = evaluate_test(test_parameter)

                        # store log file every epoch

                        store_log(log_dir, test_acc, epoch, log)

                        epoch += 1

                    epoch = 0
                    loop = 0

            count += 1

cross_valid(para_whole_dataset_dic)

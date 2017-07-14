from file_system_model import *
from basic_model import *
import flow_model as fm

#train the model
#ver 1.0
def restore_whole_dataset_begin(para_dic, model_name, model_path):

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
    [SPAN, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size, valid_file_size, test_file_size, reg, lr_rate, lr_decay, keep_prob_v, log_dir, module_dir, eval_last_num, epoch, loop] = get_para_from_dic(para_dic)

    max_step = train_file_size // batch_size
    loops = data_size // file_size
    log = Log()
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

            saver = tf.train.Saver()

            saver.restore(sess, model_path)

            while epoch < epochs:

                # show the epoch num
                words_log_print_epoch(epoch, epochs, log)

                loop_indexs = dpm.get_file_random_seq_indexs(loops)

                # caution loop is not in sequence

                while loop < loops:
                    before_time = time.time()

                    train_file = "train_set_%d.csv" % loop_indexs[loop]

                    loop_loss_v, loop_acc = do_train_file(sess, train_pl, dir, train_file, SPAN, max_step, batch_size,keep_prob_v)

                    words_log_print_loop(loop, loops, loop_loss_v, loop_acc, log)

                    # each loop_eval_num, do evaluation
                    if (loop != 0 and loop % loop_eval_num == 0) or loop == loops - 1:
                        # show the time
                        time_show(before_time, loop_eval_num, loop, loops, epoch, epochs, log)
                        # store the parameter first
                        eval_parameters = (loop, loop_indexs, SPAN, sess, batch_size, correct_num, placeholders, log)
                        # here only evaluate last eval_last_num files
                        evaluate_last_x_files(eval_last_num, eval_parameters, dir)

                        #ask for if want to interrupt
                        #press i to interrupt
                        temp_para = [SPAN, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size, valid_file_size, test_file_size, reg, lr_rate, lr_decay, keep_prob_v, log_dir, module_dir, eval_last_num, epoch, loop]

                        answer = fm.interrupt_flow(temp_para, sess, log)
                        if answer == 'Done':
                            return 'Done'

                        [SPAN, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size,valid_file_size, test_file_size, reg, lr_rate, lr_decay, keep_prob_v, log_dir, module_dir,eval_last_num, epoch, loop] = temp_para

                    loop += 1



                # each epoch decay the lr_rate
                lr_rate *= lr_decay

                # store the parameter first
                test_parameter = loops, epoch, SPAN, sess, batch_size, correct_num, placeholders, log, dir
                # do the test evaluate
                test_acc = evaluate_test(test_parameter)

                # store module every epoch
                store_module(module_dir, test_acc, epoch, sess, log)
                # store log file every epoch
                store_log(log_dir, test_acc, epoch, log)

                epoch += 1

from file_system_model import *
from basic_model import *
import flow_model as fm

#the parameter need fill
#######################################################
#from network_model_example import *
SPAN=[10]
dir = '/media/freshield/DATA_U/Ciena_new_data/20spans/norm/'
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
reg = 0.009829
lr_rate = 0.000312
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

#restore the model
#ver 1.0
def testing_result(para_dic, model_name, model_path, log_path = None, loop_indexs_path=None):

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

    loops = data_size // file_size


    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_x = tf.placeholder(tf.float32, [None, 304, 48, 2], name='input_x')
            para_pl = tf.placeholder(tf.float32, [None, 41], name='para_pl')
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

            saver = tf.train.Saver()

            saver.restore(sess, model_path)

            print ''
            print 'Model was restored'

            restore_first = True

            result_matrix = np.zeros((100*loops,6))
            label_matrix = np.zeros((100*loops,6))

            result_index = 0

            print "step",
            for test_loop in xrange(loops):
                print test_loop,
                test_file = "Raw_data_%d_test.csv" % test_loop
                X_test, para_test, y_test = dpm.prepare_dataset(dir, test_file, SPAN)
                feed_dict = {input_x: X_test, para_pl: para_test, input_y: y_test, train_phase: False,
                             keep_prob: 1.0}
                y_pred_v = sess.run(y_pred, feed_dict=feed_dict)

                result_matrix[result_index:result_index+100, :] = y_pred_v[:,:]
                label_matrix[result_index:result_index+100, :] = y_test[:,:]

                result_index += 100

            np.savetxt('result_value.csv', result_matrix, delimiter=',')
            np.savetxt('result_label.csv', label_matrix, delimiter=',')

    return 'Done'



#restore the model
#ver 1.0
def testing_and_store(para_dic, model_name, model_path, store_path, SPAN):

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
    [_, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size, valid_file_size, test_file_size, reg, lr_rate, lr_decay, keep_prob_v, log_dir, module_dir, eval_last_num, epoch, loop, best_model_number, best_model_acc_dic, best_model_dir_dic] = get_para_from_dic(para_dic)

    loops = data_size // file_size


    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_x = tf.placeholder(tf.float32, [None, 304, 48, 2], name='input_x')
            para_pl = tf.placeholder(tf.float32, [None, 41], name='para_pl')
            input_y = tf.placeholder(tf.float32, [None, 6], name='input_y')
            train_phase = tf.placeholder(tf.bool, name='train_phase')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            ENLC_array = tf.reshape(tf.constant([34.515, 23.92, 21.591, 25.829, 28.012, 29.765], dtype=tf.float32),
                                    [6, 1])

            # logits
            y_pred, parameters = model.inference(input_x, para_pl, train_phase, keep_prob)

            y_prob = tf.nn.softmax(y_pred)

            y_enlc = tf.matmul(y_prob, ENLC_array)

            y_type = tf.argmax(y_pred, 1)

            # predict
            correct_num, accuracy = corr_num_acc(input_y, y_pred)

            saver = tf.train.Saver()

            saver.restore(sess, model_path)

            print ''
            print 'Model was restored'

            restore_first = True

            result_matrix = np.zeros((100*loops,4))

            result_index = 0
            total_acc = 0.0

            print "step",
            for test_loop in xrange(loops):
                print test_loop,
                test_file = "Raw_data_%d_test.csv" % test_loop
                X_test, para_test, y_test, y_true, enlc_true = dpm.prepare_dataset_inclue_enlc(dir, test_file, SPAN)
                feed_dict = {input_x: X_test, para_pl: para_test, input_y: y_test, train_phase: False,
                             keep_prob: 1.0}
                y_type_v, y_enlc_v, acc_v = sess.run([y_type, y_enlc, accuracy], feed_dict=feed_dict)

                result_matrix[result_index:result_index+100, 0] = y_type_v
                result_matrix[result_index:result_index+100, 1] = y_true
                result_matrix[result_index:result_index+100, 2] = np.reshape(y_enlc_v, [100])
                result_matrix[result_index:result_index+100, 3] = enlc_true

                total_acc += acc_v

                result_index += 100

            total_acc /= loops
            header = 'predict_type,true_type,predict_enlc,true_enlc'
            np.savetxt(store_path + 'span%d_result_acc_%.4f.csv' % (SPAN[0],total_acc), result_matrix, delimiter=',', header=header, comments='')

            print 'span%d result acc is %.4f' % (SPAN[0],total_acc)

    return 'Done'


#restore the model
#ver 1.0
def testing_result_target_span(para_dic, model_name, model_path, store_path, SPAN, log_path = None,
                               loop_indexs_path=None):

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
    [_, dir, epochs, data_size, file_size, loop_eval_num, batch_size, train_file_size, valid_file_size,
     test_file_size, reg, lr_rate, lr_decay, keep_prob_v, log_dir, module_dir, eval_last_num, epoch, loop, best_model_number, best_model_acc_dic, best_model_dir_dic] = get_para_from_dic(para_dic)

    loops = data_size // file_size


    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_x = tf.placeholder(tf.float32, [None, 304, 48, 2], name='input_x')
            para_pl = tf.placeholder(tf.float32, [None, 41], name='para_pl')
            input_y = tf.placeholder(tf.float32, [None, 6], name='input_y')
            train_phase = tf.placeholder(tf.bool, name='train_phase')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # logits
            y_pred, parameters = model.inference(input_x, para_pl, train_phase, keep_prob)

            y_prob = tf.nn.softmax(y_pred)

            # predict
            correct_num, accuracy = corr_num_acc(input_y, y_pred)

            saver = tf.train.Saver()

            saver.restore(sess, model_path)

            print ''
            print 'Model was restored'

            restore_first = True

            result_matrix = np.zeros((100*loops,6))
            label_matrix = np.zeros((100*loops,6))
            prob_matrix = np.zeros((100*loops,6))

            result_index = 0
            total_acc = 0

            print "step",
            for test_loop in xrange(loops):
                print test_loop,
                test_file = "Raw_data_%d_test.csv" % test_loop
                X_test, para_test, y_test = dpm.prepare_dataset(dir, test_file, SPAN)
                feed_dict = {input_x: X_test, para_pl: para_test, input_y: y_test, train_phase: False,
                             keep_prob: 1.0}
                y_pred_v, acc_v, y_prob_v = sess.run([y_pred,accuracy,y_prob], feed_dict=feed_dict)

                total_acc += acc_v

                result_matrix[result_index:result_index+100, :] = y_pred_v[:,:]
                label_matrix[result_index:result_index+100, :] = y_test[:,:]
                prob_matrix[result_index:result_index+100, :] = y_prob_v[:,:]

                result_index += 100

            total_acc /= loops

            np.savetxt(store_path + 'span%d_value_acc_%.4f.csv' % (SPAN[0],total_acc), result_matrix, delimiter=',')
            np.savetxt(store_path + 'span%d_label_acc_%.4f.csv' % (SPAN[0],total_acc), label_matrix, delimiter=',')
            np.savetxt(store_path + 'span%d_prob_acc_%.4f.csv' % (SPAN[0],total_acc), prob_matrix, delimiter=',')


    return 'Done'


#para_dic = read_json_to_dic('interrupt/parameters.json')

#testing_result(para_whole_dataset_dic, 'resnet_link', '/media/freshield/New_2T_Data/corsair/CIENA/Result/modules/ciena_20spans_train/0.9292_epoch74/module.ckpt')

for i in range(20):

    span_num = i + 1

    print span_num

    model_path = '/media/freshield/DATA_U/modules/span%d/module.ckpt' % span_num

    SPAN = [span_num]

    testing_result_target_span(para_whole_dataset_dic, 'resnet_link', model_path, '/home/freshield/', SPAN)
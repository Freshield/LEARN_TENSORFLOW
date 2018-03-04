from file_system_model import *
from basic_model import *
import flow_model as fm

#the parameter need fill
#######################################################
#from network_model_example import *
SPAN=[10]
dir = '/media/freshield/New_2T_Data/Ciena_new_data/20spans/norm/'
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

#para_dic = read_json_to_dic('interrupt/parameters.json')

testing_result(para_whole_dataset_dic, 'resnet_link', '/media/freshield/New_2T_Data/corsair/CIENA/Result/modules/ciena_20spans_train/0.9292_epoch74/module.ckpt')
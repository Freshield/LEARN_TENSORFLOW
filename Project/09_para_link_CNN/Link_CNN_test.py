import time
from Link_CNN_model import *

SPAN=[10]

dir = '/media/freshield/DATA_W/Ciena_data/raw/norm/'

epochs = 100

data_size = 600000

file_size = 1000

loops = data_size // file_size

#how many loops do an evaluation
last_loop_num = 50

batch_size = 100

train_file_size = 800

valid_file_size = 100

test_file_size = 100

max_step = train_file_size // batch_size

epoch_dir = {
    10:"[==========>]",
    9: "[=========> ]",
    8: "[========>  ]",
    7: "[=======>   ]",
    6: "[======>    ]",
    5: "[=====>     ]",
    4: "[====>      ]",
    3: "[===>       ]",
    2: "[==>        ]",
    1: "[=>         ]",
    0: "[>          ]"
}

#hypers
reg = 0.00001
lr_rate = 0.001261
lr_decay = 0.99
keep_prob_v = 1.00

log = ''

log_dir = 'logs/simple_CNN_test1/'
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
            words = "\nepoch "
            words += epoch_dir[int(10 * (float(epoch) / float(epochs)))]
            words += "[%d/%d]\n" % (epoch, epochs)
            print words
            log += words + "\n"

            loop_indexs = get_file_random_seq_indexs(loops)

            #caution loop is not in sequence
            for loop in xrange(loops):
                before_time = time.time()

                train_file = "train_set_%d.csv"  % loop_indexs[loop]

                loop_loss_v, loop_acc = do_train_file(sess, train_pl, dir, train_file, SPAN, max_step, batch_size, keep_prob_v)


                words = "loop "
                words += epoch_dir[loop // (loops / 10)]
                words += "[%d/%d] " % (loop,loops)
                words += 'loss in loop %d is %f, acc is %.3f' % (loop, loop_loss_v, loop_acc)
                print words
                log += words + "\n"

                #each 50 loop, do evaluation
                #!!!!!!!!!!!!!!!caution, loop from 0-49, but % 50 == 0, maybe have problem
                if (loop != 0 and loop % last_loop_num == 0) or loop == loops - 1:
                    #show the time
                    time_show(before_time, last_loop_num, loop, loops, epoch, epochs, log)

                    #do the evaluation for the last 10 files
                    #last 100 files are too slow
                    train_acc = 0.0
                    valid_acc = 0.0
                    print "step",
                    for step in xrange(10):
                        print step,
                        #careful for the file name
                        train_file = "train_set_%d.csv" % loop_indexs[loop - 10 + step]
                        validation_file = "validation_set_%d.csv" % loop_indexs[loop - 10 + step]

                        X_train, para_train, y_train = prepare_dataset(dir, train_file, SPAN)
                        X_valid, para_valid, y_valid = prepare_dataset(dir, validation_file, SPAN)

                        step_train_acc = do_eval(sess, X_train, para_train, y_train, batch_size, correct_num, placeholders)
                        train_acc += step_train_acc

                        step_valid_acc = do_eval(sess, X_valid, para_valid, y_valid, batch_size, correct_num, placeholders)
                        valid_acc += step_valid_acc

                    train_acc /= 10
                    valid_acc /= 10
                    print "\n",
                    print ('----------train acc in loop %d is %.4f----------' % (loop, train_acc))
                    log += ('----------train acc in loop %d is %.4f----------\n' % (loop, train_acc))
                    print ('----------valid acc in loop %d is %.4f----------' % (loop, valid_acc))
                    log += ('----------valid acc in loop %d is %.4f----------\n' % (loop, valid_acc))


            #each epoch decay the lr_rate
            lr_rate *= lr_decay



            #each epoch do a test evaluation
            test_acc = 0.0
            print "step",
            for test_loop in xrange(loops):
                print test_loop,
                test_file = "test_set_%d.csv" % test_loop
                X_test, para_test, y_test = prepare_dataset(dir, test_file, SPAN)
                loop_test_acc = do_eval(sess, X_test, para_test, y_test, batch_size, correct_num, placeholders)
                test_acc += loop_test_acc
            test_acc /= loops
            print ""
            print ('----------epoch %d test accuracy is %f----------' % (epoch,test_acc))
            log += ('----------epoch %d test accuracy is %f----------\n' % (epoch,test_acc))

            filename = log_dir + '%.4f_epoch%d' % (test_acc, epoch)
            f = file(filename, 'w+')
            f.write(log)
            f.close()




import time
from CNN_model import *

SPAN=[10]

dir = '/media/freshield/LINUX/Ciena/CIENA/raw/norm/'

epochs = 100

loops = 600

batch_size = 100

train_file_size = 800

valid_file_size = 100

test_file_size = 100

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
reg = 0.000067
lr_rate = 0.002
lr_decay = 0.99
keep_prob_v = 0.9569

log = ''

log_dir = 'logs/simple_CNN_test1/'
del_and_create_dir(log_dir)

with tf.Graph().as_default():
    with tf.Session() as sess:
        #inputs
        input_x = tf.placeholder(tf.float32, [None, 32, 104, 3], name='input_x')
        input_y = tf.placeholder(tf.float32, [None, 3], name='input_y')
        train_phase = tf.placeholder(tf.bool, name='train_phase')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        #logits
        y_pred, parameters = inference(input_x, train_phase, keep_prob)

        #loss
        loss_value = loss(input_y, y_pred, reg, parameters)

        #train
        train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_value)

        #predict
        correct_num, accuracy = corr_num_acc(input_y, y_pred)

        #placeholders
        placeholders = (input_x, input_y, train_phase, keep_prob)

        sess.run(tf.global_variables_initializer())

        for epoch in xrange(epochs):

            #show the epoch num
            words = "\nepoch "
            words += epoch_dir[epoch // (epochs / 10)]
            words += "[%d/%d]\n" % (epoch, epochs)
            print words
            log += words

            loop_indexs = get_file_random_seq_indexs(loops)

            #caution loop is not in sequence
            for loop in xrange(loops):
                before_time = time.time()
                train_file = "train_set_%d.csv"  % loop_indexs[loop]

                X_train, y_train = prepare_dataset(dir,train_file,SPAN)

                indexs = get_random_seq_indexs(X_train)
                out_of_dataset = False
                last_index = 0

                loop_loss_v = 0.0
                loop_acc = 0.0

                max_step = train_file_size // batch_size

                #one loop, namely, one file
                for step in xrange(max_step):

                    #should not happen
                    if out_of_dataset == True:
                        print "out of dataset"
                        indexs = get_random_seq_indexs(X_train)
                        last_index = 0
                        out_of_dataset = False

                    last_index, data, out_of_dataset = sequence_get_data(X_train, y_train, indexs, last_index,
                                                                         batch_size)

                    feed_dict = {input_x: data['X'], input_y: data['y'], train_phase: True, keep_prob:keep_prob_v}
                    _, loss_v, acc = sess.run([train_step, loss_value, accuracy], feed_dict=feed_dict)

                    loop_loss_v += loss_v
                    loop_acc += acc

                loop_loss_v /= max_step
                loop_acc /= max_step

                words = "loop "
                words += epoch_dir[loop // (loops / 10)]
                words += "[%d/%d] " % (loop,loops)
                words += 'loss in loop %d is %f, acc is %.3f' % (loop, loop_loss_v, loop_acc)
                print words
                log += words

                #each 100 loop, do evaluation
                #!!!!!!!!!!!!!!!caution, loop from 0-99, but % 100 == 0, maybe have problem
                if (loop != 0 and loop % 30 == 0) or loop == loops - 1:
                    #show the time
                    last_time = time.time()
                    span_time = last_time - before_time
                    print ('last 100 loop use %f minutes' % (span_time * 100 / 60))
                    print ('rest loop need %.3f minutes' % (span_time * (loops - loop) / 60))
                    print ('rest epoch need %.3f hours' % (span_time * (loops - loop) * (epochs - epoch) / 3600))
                    log += ('last 100 loop use %f minutes\n' % (span_time * 100 / 60))
                    log += ('rest loop need %.3f minutes\n' % (span_time * (loops - loop) / 60))
                    log += ('rest epoch need %.3f hours' % (span_time * (loops - loop) * (epochs - epoch) / 3600))

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

                        X_train, y_train = prepare_dataset(dir, train_file, SPAN)
                        X_valid, y_valid = prepare_dataset(dir, validation_file, SPAN)

                        step_train_acc = do_eval(sess, X_train, y_train, batch_size, correct_num, placeholders)
                        train_acc += step_train_acc

                        step_valid_acc = do_eval(sess, X_valid, y_valid, batch_size, correct_num, placeholders)
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
            for loop in xrange(loops):
                test_file = "test_set_%d.csv" % loop
                X_test, y_test = prepare_dataset(dir, test_file, SPAN)
                loop_test_acc = do_eval(sess, X_test, y_test, batch_size, correct_num, placeholders)
                test_acc += loop_test_acc
            test_acc /= loops
            print ('----------epoch %d test accuracy is %f----------' % (test_acc))
            log += ('----------epoch %d test accuracy is %f----------\n' % (test_acc))


        filename = log_dir + 'epoch%d' % epoch
        f = file(filename, 'w+')
        f.write(log)
        f.close()

import time
from Link_CNN_model import *

SPAN=[20]

dir = '/home/freshield/Ciena_data/dataset_10k/'

epochs = 20

data_size = 10000

file_size = 10000

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

log_dir = 'logs/Link_CNN_10k'
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

            before_time = time.time()

            train_file = "train_set_norm.csv"

            loop_loss_v, loop_acc = do_train_file(sess, train_pl, dir, train_file, SPAN, max_step, batch_size,keep_prob_v, log)


            # show the time
            time_show_10k(before_time, epoch, epochs, log)

            train_acc = 0.0
            valid_acc = 0.0

            # careful for the file name
            train_file = "train_set_norm.csv"
            validation_file = "validation_set_norm.csv"

            X_train, para_train, y_train = prepare_dataset(dir, train_file, SPAN)
            X_valid, para_valid, y_valid = prepare_dataset(dir, validation_file, SPAN)

            train_acc = do_eval(sess, X_train, para_train, y_train, batch_size, correct_num, placeholders)

            valid_acc = do_eval(sess, X_valid, para_valid, y_valid, batch_size, correct_num, placeholders)

            print "\n",
            print ('----------train acc is %.4f----------' % (train_acc))
            log += ('----------train acc is %.4f----------\n' % (train_acc))
            print ('----------valid acc is %.4f----------' % (valid_acc))
            log += ('----------valid acc is %.4f----------\n' % (valid_acc))


            #each epoch decay the lr_rate
            lr_rate *= lr_decay



            #each epoch do a test evaluation
            test_acc = 0.0

            test_file = "test_set_norm.csv"
            X_test, para_test, y_test = prepare_dataset(dir, test_file, SPAN)
            test_acc = do_eval(sess, X_test, para_test, y_test, batch_size, correct_num, placeholders)

            print ""
            print ('----------epoch %d test accuracy is %f----------' % (epoch,test_acc))
            log += ('----------epoch %d test accuracy is %f----------\n' % (epoch,test_acc))

            filename = log_dir + '%.4f_epoch%d' % (test_acc, epoch)
            f = file(filename, 'w+')
            f.write(log)
            f.close()

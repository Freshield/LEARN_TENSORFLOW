import pandas as pd
import time
from resnet_model import *

NROWS = 10000

SPAN=[10]

filename = '~/Ciena_data/ciena10000.csv'

data = pd.read_csv(filename, header=None, nrows=NROWS)

print 'Data Shape: %s' % str(data.shape)

train_set, validation_set, test_set = split_dataset(data, radio=0.1)

#normalize dataset
train_set, train_min, train_max = normalize_dataset(train_set)
validation_set, _, _ = normalize_dataset(validation_set, train_min, train_max)
test_set, _, _ = normalize_dataset(test_set, train_min, train_max)

#reshape dataset
X_train, y_train = reshape_dataset(train_set, SPAN)
X_valid, y_valid = reshape_dataset(validation_set, SPAN)
X_test, y_test = reshape_dataset(test_set, SPAN)

#hypers
reg = 1e-4
lr_rate = 0.002
max_step = 30000
batch_size = 100
lr_decay = 0.99
lr_epoch = 800

log = ''

with tf.Graph().as_default():
    with tf.Session() as sess:
        #inputs
        input_x = tf.placeholder(tf.float32, [None, 32, 100, 3], name='input_x')
        input_y = tf.placeholder(tf.float32, [None, 3], name='input_y')
        train_phase = tf.placeholder(tf.bool, name='train_phase')

        #logits
        y_pred, parameters = inference(input_x, train_phase)

        #loss
        loss_value = loss(input_y, y_pred, reg, parameters)

        #train
        train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_value)

        #predict
        correct_num, accuracy = corr_num_acc(input_y, y_pred)

        #placeholders
        placeholders = (input_x, input_y, train_phase)

        sess.run(tf.global_variables_initializer())

        indexs = get_random_seq_indexs(X_train)
        out_of_dataset = False
        last_index = 0

        for step in xrange(max_step):
            before_time = time.time()

            if out_of_dataset == True:
                indexs = get_random_seq_indexs(X_train)
                last_index = 0
                out_of_dataset = False

            last_index, data, out_of_dataset = sequence_get_data(X_train, y_train, indexs, last_index, batch_size)

            feed_dict = {input_x:data['X'], input_y:data['y'], train_phase:True}
            _, loss_v, acc = sess.run([train_step, loss_value, accuracy], feed_dict=feed_dict)

            if step % 50 == 0:
                print 'loss in step %d is %f, acc is %.3f' % (step, loss_v, acc)
                log += 'loss in step %d is %f, acc is %.3f\n' % (step, loss_v, acc)

            if step % 300 == 0 or step == max_step - 1:
                last_time = time.time()
                span_time = last_time - before_time
                print ('last 300 loop use %f minutes' % (span_time * 300 / 60))
                print ('rest time is %.3f minutes' % (span_time * (max_step - step) / 60))
                log += ('last 300 loop use %f minutes\n' % (span_time * 300 / 60))
                log += ('rest time is %.3f minutes\n' % (span_time * (max_step - step) / 60))

                result = do_eval(sess, X_train, y_train, batch_size, correct_num, placeholders)
                print ('----------train acc in step %d is %.4f----------' % (step, result))
                log += ('----------train acc in step %d is %.4f----------\n' % (step, result))

                result = do_eval(sess, X_valid, y_valid, batch_size, correct_num, placeholders)
                print ('----------valid acc in step %d is %.4f----------' % (step, result))
                log += ('----------valid acc in step %d is %.4f----------\n' % (step, result))

            if step > 0 and step % lr_epoch == 0:
                lr_rate *= lr_decay

        result = do_eval(sess, X_test, y_test, batch_size, correct_num, placeholders)
        print ('----------last accuracy is %f----------' % (result))
        log += ('----------last accuracy is %f----------\n' % (result))

filename = 'resnet_test1_log'
f = file(filename, 'w+')
f.write(log)
f.close()

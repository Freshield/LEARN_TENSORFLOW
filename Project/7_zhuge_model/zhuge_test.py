import pandas as pd
import time
from zhuge_nn_model import *

NROWS = 10000

SPAN=[19]

log = ''

filename = '/home/freshield/Ciena_data/ciena_pca_10000.csv'

data = pd.read_csv(filename, header=None, nrows=NROWS)

words = 'Data Shape: %s' % str(data.shape)
print words
log += words + '\n'

train_set, validation_set, test_set = split_dataset(data, radio=0.1)

#normalize dataset
train_set, train_mean = normalize_dataset(train_set)
validation_set, _ = normalize_dataset(validation_set, train_mean)
test_set, _ = normalize_dataset(test_set, train_mean)

X_train, y_train = reshape_dataset(train_set, SPAN)
X_valid, y_valid = reshape_dataset(validation_set, SPAN)
X_test, y_test = reshape_dataset(test_set, SPAN)

print X_train.shape

#hypers
lr_rate = 0.002
max_step = 30000
batch_size = 100
lr_decay = 0.99
lr_epoch = 1000

with tf.Graph().as_default():
    with tf.Session() as sess:
        #inputs
        input_x = tf.placeholder(tf.float32, [None, 241], name='input_x')
        input_y = tf.placeholder(tf.float32, [None, 3], name='input_y')

        #logits
        y_pred, parameters = inference(input_x, tf.nn.relu)

        #loss
        loss_value = loss(input_y, y_pred)

        #train
        train_step = tf.train.AdamOptimizer(lr_rate).minimize(loss_value)

        #predict
        correct_num, accuracy = corr_num_acc(input_y, y_pred)

        # placeholders
        placeholders = (input_x, input_y)

        sess.run(tf.global_variables_initializer())

        #prepare to train
        indexs = get_random_seq_indexs(X_train)
        out_of_dataset = False
        last_index = 0

        for step in xrange(max_step):
            before_time = time.time()

            if out_of_dataset == True:
                indexs = get_random_seq_indexs(X_train)
                last_index = 0
                out_of_dataset = False

            #get batch data
            last_index, data, out_of_dataset = sequence_get_data(X_train, y_train, indexs, last_index, batch_size)

            feed_dict = {input_x:data['X'], input_y:data['y']}

            #train
            _, loss_v, acc = sess.run([train_step, loss_value, accuracy], feed_dict=feed_dict)

            #show loss and batch acc
            if step % 100 == 0:
                words = 'loss in step %d is %f, acc is %.3f' % (step, loss_v, acc)
                print words
                log += words + '\n'

            #do evaluation
            if step % 500 == 0 or step == max_step - 1:
                last_time = time.time()
                span_time = last_time - before_time
                words = 'last 300 steps use %f sec\n' % (span_time * 500)
                words += 'rest time is %.3f minutes' % (span_time * (max_step - step) / 60)
                print words
                log += words + '\n'

                result = do_eval(sess, X_train, y_train, batch_size, correct_num, placeholders)
                words = '----------train acc in step %d is %.4f----------' % (step ,result)
                print words
                log += words + '\n'

                result = do_eval(sess, X_valid, y_valid, batch_size, correct_num, placeholders)
                words = '----------valid acc in step %d is %.4f----------' % (step, result)
                print words
                log += words + '\n'

            #learning rate decay
            if step > 0 and step % lr_epoch == 0:
                lr_rate *= lr_decay

        #final test
        result = do_eval(sess, X_test, y_test, batch_size, correct_num, placeholders)
        words = '----------last accuracy is %f----------' % (result)
        print words
        log += words + '\n'

#write file
filename = 'zhuge_test1_log_%d' % (SPAN[0])
f = file(filename, 'w+')
f.write(log)
f.close()















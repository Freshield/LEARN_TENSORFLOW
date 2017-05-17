import tensorflow as tf

import modules as m
#global variable
learning_rate = 0.01
max_steps = 3000
hidden1 = 128
hidden2 = 32
batch_size = 100
reg = 0.0001

#from modules give the handle
import tensorflow.examples.tutorials.mnist.input_data as input_data

data_sets = input_data.read_data_sets('MNIST')

for lr in [0.01, 0.001, 0.005, 0.002]:

    with tf.Graph().as_default():
        images_pl, labels_pl = m.placeholder_inputs(batch_size)

        logits, inf_cache = m.inference(images_pl, hidden1, hidden2)

        correct_nums = m.evaluation(logits, labels_pl)

        loss_h = m.loss(logits, labels_pl, inf_cache, reg)

        sess = tf.Session()

        train_op = m.training(loss_h, lr)

        ######################for tensorboard#############################
        log_dir = 'logs/fully_connected_net_scope/lr' + str(lr)
        #######################for tensorboard logs##############################3
        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)
        tf.summary.scalar('loss' + str(lr), loss_h)
        tf.summary.scalar('loss', loss_h)
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        for step in xrange(max_steps):

            feed_dict = m.fill_feed_dict(data_sets.train, images_pl, labels_pl, batch_size)

            _, loss = sess.run([train_op, loss_h], feed_dict=feed_dict)

            if step % 100 == 0 or step == max_steps - 1:
                print ('lr %.4f, step %d: loss = %.3f' % (lr, step, loss))

                ##########################for tensorboard########################
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) == max_steps:
                print ('Training Eval:')
                m.do_eval(sess, correct_nums, images_pl, labels_pl, data_sets.train, batch_size)

                print ('Validation Eval:')
                m.do_eval(sess, correct_nums, images_pl, labels_pl, data_sets.validation, batch_size)

                print ('Test Eval:')
                m.do_eval(sess, correct_nums, images_pl, labels_pl, data_sets.test, batch_size)

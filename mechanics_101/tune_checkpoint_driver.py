import tensorflow as tf

import modules as m
import os
#global variable
#learning_rate = 0.01
max_steps = 10000
#hidden1 = 128
#hidden2 = 32
#batch_size = 100
#reg = 2e-3

best_val = -1

#choices
batch_choices = [20, 50, 100]
lr_choices = [5e-4, 1e-3, 5e-3, 2e-3, 1e-2]
hidden1_choices = [x * 40 for x in range(1, 5, 1)]
hidden2_choices = [x * 40 for x in range(1, 5, 1)]
reg_choices = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]

#the hyperparamers
learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')

reg_str_pl = tf.placeholder(tf.float32, name='reg_str')

#from modules give the handle
import tensorflow.examples.tutorials.mnist.input_data as input_data

data_sets = input_data.read_data_sets('MNIST')

checkpoint_file = 'logs/checkpoint_model/model.ckpt'

sess = tf.Session()

total_times = len(batch_choices) * len(hidden2_choices) * len(hidden1_choices) * len(reg_choices) * len(lr_choices)

for batch_size in batch_choices:
    for hidden1 in hidden1_choices:
        for hidden2 in hidden2_choices:

            images_pl, labels_pl = m.placeholder_inputs(batch_size)

            logits, inf_cache = m.inference(images_pl, hidden1, hidden2)

            correct_nums = m.evaluation(logits, labels_pl)

            # begin train

            loss_h = m.loss(logits, labels_pl, inf_cache, reg_str_pl)

            train_op = m.training(loss_h, learning_rate_pl)

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()

            saver.save(sess, checkpoint_file, global_step=0)

            for lr in lr_choices:
                for reg in reg_choices:
                    """
                                    log_dir = 'logs/checkpoint' + str(lr)
                                    #######################for tensorboard logs##############################
                                    if tf.gfile.Exists(log_dir):
                                        tf.gfile.DeleteRecursively(log_dir)
                                    tf.gfile.MakeDirs(log_dir)
                                    ######################for tensorboard#############################
                                    tf.summary.scalar('loss' + str(lr), loss_h)
                                    summary = tf.summary.merge_all()
                                    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
                                    """
                    total_times -= 1

                    ori_lr = lr + 0

                    print
                    print 'rest times is: ', total_times
                    print 'batch size is: ', batch_size
                    print 'hidden1 size is: ', hidden1
                    print 'hidden2 size is: ', hidden2
                    print 'reg size is: ', reg
                    print 'learning rate is: ', ori_lr

                    saver.restore(sess, checkpoint_file + '-0')

                    for step in xrange(max_steps):

                        feed_dict = m.fill_feed_dict(data_sets.train, images_pl, labels_pl, batch_size)

                        feed_dict[learning_rate_pl] = lr

                        feed_dict[reg_str_pl] = reg

                        _, loss = sess.run([train_op, loss_h], feed_dict=feed_dict)
                        """
                        if step % 100 == 0 or step == max_steps - 1:
                            ##########################for tensorboard########################
                            summary_str = sess.run(summary, feed_dict=feed_dict)
                            summary_writer.add_summary(summary_str, step)
                            summary_writer.flush()
                        """
                        if step % (max_steps / batch_size) == 0:
                            lr *= 0.97

                        #if step % 1000 == 0 or step == max_steps - 1:
                        #    print ('lr %.8f, step %d: loss = %.3f' % (lr, step, loss))

                        if (step + 1) == max_steps:
                            #print ('Training Eval:')
                            #m.do_eval(sess, correct_nums, images_pl, labels_pl, data_sets.train, batch_size)

                            print ('Validation Eval:')
                            val_pre = m.do_eval(sess, correct_nums, images_pl, labels_pl, data_sets.validation, batch_size)

                            print ('Test Eval:')
                            test_pre = m.do_eval(sess, correct_nums, images_pl, labels_pl, data_sets.test, batch_size)

                            #for save temp file
                            filename = 'files/loop_%d_val_%.2f_test_%.2f.txt' % (total_times, val_pre, test_pre)

                            content = "now val_pre is:" + str(val_pre) + '\n' + \
                                      "now test_pre is:" + str(test_pre) + '\n' + \
                                      "now rest times is:" + str(total_times) + '\n' + \
                                      "now hidden1 size is: " + str(hidden1) + '\n' + \
                                      "now hidden2 size is: " + str(hidden2) + '\n' + \
                                      "now learning rate is: " + str(ori_lr) + '\n' + \
                                      "now regualrization streagths is: " + str(reg) + '\n' + \
                                      "now batch size is: " + str(batch_size)

                            test_file = open(filename, 'w')

                            test_file.write(content)

                            test_file.close()

                            #get the best model
                            if test_pre > best_val:
                                best_val = test_pre

                                print "=============================================="
                                print "best accuray is:", best_val
                                print "best hidden1 size is: ", hidden1
                                print "best hidden2 size is: ", hidden2
                                print "best learning rate is: ", ori_lr
                                print "best regualrization streagths is: ", reg
                                print "best batch size is: ", batch_size
                                print "=============================================="

                                filename = 'files/loop_%d_!!!!!!BEST!!!!!!_acc_%.2f.txt' % (total_times, best_val)

                                content = "best accuray is:" + str(best_val) + '\n' + \
                                          "best rest times is:" + str(total_times) + '\n' + \
                                          "best hidden1 size is: " + str(hidden1) + '\n' + \
                                          "best hidden2 size is: " + str(hidden2) + '\n' + \
                                          "best learning rate is: " + str(ori_lr) + '\n' + \
                                          "best regualrization streagths is: " + str(reg) + '\n' + \
                                          "best batch size is: " + str(batch_size)

                                test_file = open(filename, 'w')

                                test_file.write(content)

                                test_file.close()

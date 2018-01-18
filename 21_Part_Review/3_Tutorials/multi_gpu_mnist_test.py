#coding=utf-8

import sys
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from multi_gpu_mnist_test_model import *

#创建一个空类，底下把各种资料都存到类中
class PARAMETERS(object):
    pass


BATCH_SIZE = 128

log_device_placement = False

REG = 0.0001

EPOCH_NUM = 5



def multi_gpu(NUM_GPU):
    PARA = PARAMETERS()
    PARA.BATCH_SIZE = BATCH_SIZE
    PARA.REG = REG
    PARA.EPOCH_NUM = EPOCH_NUM
    PARA.NUM_GPU = NUM_GPU


    #从数据集获取两倍的batch
    batch_size = BATCH_SIZE * NUM_GPU



    #获取mnist的输入
    mnist = input_data.read_data_sets('data/mnist',one_hot=True)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        #lr作为一个可变的pl
        LEARNING_RATE = tf.placeholder(tf.float32, shape=[])
        PARA.LEARNING_RATE = LEARNING_RATE

        # 每个GPU模型的输入pl
        images = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.float32, [None, 10])

        apply_gradient_op, aver_loss_op, aver_acc_op = tower_model(
            images, labels, PARA
        )

        # 选择是否显示每个op和varible的物理位置
        #允许分配位置
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=log_device_placement)
        # 让gpu模式为随取随用而不是直接全部占满
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            print('run train op...')
            #初始化variable
            sess.run(tf.global_variables_initializer())
            #设置lr
            lr = 0.01
            for epoch in range(EPOCH_NUM):

                ###########################################
                # TRAINING PART
                ###########################################

                start_time = time.time()
                #一个epoch需要的batch循环次数
                total_batch = mnist.train.num_examples // batch_size
                avg_loss = 0.0
                avg_acc = 0.0
                print('\n---------------------')
                print('Epoch:%d, lr:%.4f' % (epoch, lr))
                #一个epoch的循环
                for batch_idx in range(total_batch):
                    #得到输入batch
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    #运行时输入的数据
                    feed_dict = {}
                    feed_dict[LEARNING_RATE] = lr
                    feed_dict[images] = batch_x
                    feed_dict[labels] = batch_y

                    _, _loss, _acc = sess.run([apply_gradient_op, aver_loss_op, aver_acc_op], feed_dict=feed_dict)
                    #得到每步的loss
                    avg_loss += _loss
                    #得到每步的acc
                    avg_acc += _acc
                #平均一个epoch的training loss
                avg_loss /= total_batch
                #平均一个epoch的training acc
                avg_acc /= total_batch

                #学习速度衰减
                lr = max(lr * 0.9, 0.00001)


                print('Train loss:%.4f' % (avg_loss))
                print('Train acc:%.4f' % (avg_acc))

                #得到当前epoch的训练时间
                stop_time = time.time()

                elapsed_time = stop_time - start_time
                print('Cost time: ' + str(elapsed_time) + ' sec.')

                #显示当前GPU用量
                gpu_info = os.popen('nvidia-smi')
                print(gpu_info.read())

                ###########################################
                # VALIDATION PART
                ###########################################
                #得到validation的一个epoch需要的循环次数
                total_batch = mnist.validation.num_examples // batch_size
                val_acc = 0.0

                #开始循环做validation
                for batch_idx in range(total_batch):
                    #得到当前batch的数据
                    batch_x, batch_y = mnist.validation.next_batch(batch_size)
                    #运行时输入的数据
                    feed_dict = {}
                    feed_dict[images] = batch_x
                    feed_dict[labels] = batch_y
                    #得到当前batch的accuracy
                    _acc = sess.run(aver_acc_op, feed_dict=feed_dict)
                    val_acc += _acc
                #算出平均的validation accuracy
                val_accuracy = val_acc / total_batch
                print('Val Accuracy: %0.4f%%' % (100.0 * val_accuracy))

            #完成一个epoch的训练
            print('training done.')

            ###########################################
            #TESTING PART
            ###########################################
            #得到testing需要的batch循环次数
            total_batch = mnist.test.num_examples // batch_size
            test_acc = 0.0

            #开始循环batch来testing
            for batch_idx in range(total_batch):
                #得到这个batch的testing数据
                batch_x, batch_y = mnist.test.next_batch(batch_size)
                #运行时输入数据
                feed_dict = {}
                feed_dict[images] = batch_x
                feed_dict[labels] = batch_y
                #得到当前batch的accuracy
                _acc = sess.run(aver_acc_op, feed_dict=feed_dict)
                test_acc += _acc
            #得到平均的accuracy
            test_accuracy = test_acc / total_batch
            print('Test Accuracy: %0.4f%%\n\n' % (100.0 * test_accuracy))


def single_gpu():
    batch_size = 128
    mnist = input_data.read_data_sets('data/mnist',one_hot=True)

    tf.reset_default_graph()

    # 选择是否显示每个op和varible的物理位置
    config = tf.ConfigProto(log_device_placement=log_device_placement)
    # 让gpu模式为随取随用而不是直接全部占满
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        #lr作为一个可变的pl
        learning_rate = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        print('build model...')
        print('build model on gpu tower...')
        # 每个GPU模型的输入pl
        images = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.float32, [None, 10])
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                # 得到loss和accuracy
                loss, acc = build_model(images, labels, REG)

                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(loss)
        aver_loss_op = loss
        apply_gradient_op = opt.apply_gradients(grads)
        aver_acc_op = tf.reduce_mean(acc)

        # 选择是否显示每个op和varible的物理位置
        config = tf.ConfigProto(log_device_placement=log_device_placement)
        # 让gpu模式为随取随用而不是直接全部占满
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            print('run train op...')
            sess.run(tf.global_variables_initializer())
            lr = 0.01
            for epoch in range(EPOCH_NUM):

                ###########################################
                # TRAINING PART
                ###########################################

                start_time = time.time()
                total_batch = mnist.train.num_examples // batch_size
                avg_loss = 0.0
                avg_acc = 0.0
                print('\n---------------------')
                print('Epoch:%d, lr:%.4f' % (epoch, lr))
                for batch_idx in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    inp_dict = {}
                    inp_dict[learning_rate] = lr
                    inp_dict[images] = batch_x
                    inp_dict[labels] = batch_y
                    _, _loss, _acc = sess.run([apply_gradient_op, aver_loss_op, aver_acc_op], inp_dict)
                    avg_loss += _loss
                    avg_acc += _acc
                avg_loss /= total_batch
                avg_acc /= total_batch

                lr = max(lr * 0.7, 0.00001)


                print('Train loss:%.4f' % (avg_loss))
                print('Train acc:%.4f' % (avg_acc))

                gpu_info = os.popen('nvidia-smi')
                print(gpu_info.read())

                ###########################################
                # VALIDATION PART
                ###########################################
                #val_payload_per_gpu = batch_size // num_gpu
                total_batch = int(mnist.validation.num_examples / batch_size)
                val_acc = 0.0

                for batch_idx in range(total_batch):
                    batch_x, batch_y = mnist.validation.next_batch(batch_size)
                    inp_dict[images] = batch_x
                    inp_dict[labels] = batch_y
                    _acc = sess.run(aver_acc_op, inp_dict)
                    val_acc += _acc
                val_accuracy = val_acc / total_batch
                print('Val Accuracy: %0.4f%%' % (100.0 * val_accuracy))

                stop_time = time.time()
                elapsed_time = stop_time - start_time
                print('Cost time: ' + str(elapsed_time) + ' sec.')

            print('training done.')

            ###########################################
            #TESTING PART
            ###########################################
            #test_payload_per_gpu = batch_size // num_gpu
            total_batch = int(mnist.test.num_examples / batch_size)
            test_acc = 0.0

            for batch_idx in range(total_batch):
                batch_x, batch_y = mnist.test.next_batch(batch_size)
                inp_dict[images] = batch_x
                inp_dict[labels] = batch_y
                _acc = sess.run(aver_acc_op, inp_dict)
                test_acc += _acc
            test_accuracy = test_acc / total_batch
            print('Test Accuracy: %0.4f%%\n\n' % (100.0 * test_accuracy))

if __name__ == '__main__':
    #single_gpu()
    multi_gpu(2)
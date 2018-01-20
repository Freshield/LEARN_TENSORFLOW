import sys
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from multi_gpu_mnist_test_model import *

#Test helper
#if LOOK_FLAG is True, then will print the values
LOOK_FLAG = False

def look_value(*args):
    if type(args[-1]) is bool:
        flag = args[-1]
    else:
        flag = LOOK_FLAG

    if flag:
        print(args)

BATCH_SIZE = 128
log_device_placement = False
reg = 0.0001

def average_gradients(tower_grads):
    print('average_gradients')
    average_grads = []
    #tower_grads构成如下
    #([(tower0.conv1.grads,tower0.conv1),(tower0.bias1.grads,tower0.bias1)...],
    # [(tower1.conv1.grads,tower1.conv1),(tower1.bias1.grads,tower1.bias1)...])
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        #比如第一个就是((tower0_conv1_grads,tower0_conv1),(tower1_conv1_grads,tower1_conv1))

        #grads相当于我只取前边的grads
        #比如第一个就是
        #[tower0_conv1_grads,tower1_conv1_grads]
        grads = [g for g, _ in grad_and_vars]

        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        #因为我们共享权重，所以只需要返回一个tower的权重就可以了
        #这里相当于我们只取第一个tower的权重
        v = grad_and_vars[0][1]
        #这里的tuple是(平均的grads,variables)
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
        #最后averages相当于
        #[(avg_conv1.grads,conv1),(avg_bias1.grads,bias1),...]
    return average_grads

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
        # model数组为每个gpu的tuple数组
        models = []
        # 每个GPU模型的输入pl
        images = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.float32, [None, 10])
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                # 得到pred, 和loss
                pred, loss, acc = build_model(images, labels, reg)

                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(loss)
                models.append((images, labels, pred, loss, grads, acc))
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
            for epoch in range(5):

                ###########################################
                # TRAINING PART
                ###########################################

                start_time = time.time()
                #payload_per_gpu = batch_size // num_gpu
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
                    #inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y)
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
                    #inp_dict = feed_all_gpu({}, models, val_payload_per_gpu, batch_x, batch_y)
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
                #inp_dict = feed_all_gpu({}, models, test_payload_per_gpu, batch_x, batch_y)
                inp_dict[images] = batch_x
                inp_dict[labels] = batch_y
                _acc = sess.run(aver_acc_op, inp_dict)
                test_acc += _acc
            test_accuracy = test_acc / total_batch
            print('Test Accuracy: %0.4f%%\n\n' % (100.0 * test_accuracy))

def multi_gpu(num_gpu):
    #从数据集获取两倍的batch
    batch_size = BATCH_SIZE * num_gpu

    mnist = input_data.read_data_sets('data/mnist',one_hot=True)

    tf.reset_default_graph()

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        #lr作为一个可变的pl
        learning_rate = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        print('build model...')
        print('build model on gpu tower...')
        # model数组为每个gpu的tuple数组
        models = []
        # 每个GPU模型的输入pl
        images = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.float32, [None, 10])
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_id in range(num_gpu):
                with tf.device('/gpu:%d' % gpu_id):
                    print('tower:%d...' % gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):

                        if gpu_id == 0:
                            x = images[:BATCH_SIZE]
                            y = labels[:BATCH_SIZE]
                        else:
                            x = images[BATCH_SIZE:]
                            y = labels[BATCH_SIZE:]

                        #得到pred, 和loss
                        pred, loss, acc = build_model(x, y, reg)

                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(loss)
                        models.append((x, y, pred, loss, grads, acc))

        print('build model on gpu tower done.')

        print('reduce model on cpu...')
        tower_x, tower_y, tower_preds, tower_losses, tower_grads, tower_acc = zip(*models)
        aver_loss_op = tf.reduce_mean(tower_losses)
        apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))
        aver_acc_op = tf.reduce_mean(tower_acc)

        # 选择是否显示每个op和varible的物理位置
        config = tf.ConfigProto(log_device_placement=log_device_placement)
        # 让gpu模式为随取随用而不是直接全部占满
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            print('run train op...')
            sess.run(tf.global_variables_initializer())
            lr = 0.01
            for epoch in range(5):

                ###########################################
                # TRAINING PART
                ###########################################

                start_time = time.time()
                #payload_per_gpu = batch_size // num_gpu
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
                    #inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y)
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
                    #inp_dict = feed_all_gpu({}, models, val_payload_per_gpu, batch_x, batch_y)
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
                #inp_dict = feed_all_gpu({}, models, test_payload_per_gpu, batch_x, batch_y)
                inp_dict[images] = batch_x
                inp_dict[labels] = batch_y
                _acc = sess.run(aver_acc_op, inp_dict)
                test_acc += _acc
            test_accuracy = test_acc / total_batch
            print('Test Accuracy: %0.4f%%\n\n' % (100.0 * test_accuracy))

if __name__ == '__main__':
    #single_gpu()
    multi_gpu(2)
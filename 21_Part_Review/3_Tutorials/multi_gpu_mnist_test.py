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
REG = 0.0001
EPOCH_NUM = 5

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
        #因为我们的grads也是数组，这里用stack把同个variable不同tower
        #的grads立起来，堆到一起，再竖着计算
        #[[tower0_conv1_grads],
        # [tower1_conv1_grads]]
        grad = tf.stack(grads, 0)
        #这里竖着求出mean
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

def multi_gpu(num_gpu):
    #从数据集获取两倍的batch
    batch_size = BATCH_SIZE * num_gpu

    #获取mnist的输入
    mnist = input_data.read_data_sets('data/mnist',one_hot=True)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        #lr作为一个可变的pl
        learning_rate = tf.placeholder(tf.float32, shape=[])
        #获取optimizer
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        print('build model...')
        print('build model on gpu tower...')
        # model数组为每个gpu的tuple数组
        models = []
        # 每个GPU模型的输入pl
        images = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.float32, [None, 10])
        #这里来做不同的GPU的tower模型
        with tf.variable_scope(tf.get_variable_scope()):
            #获取gpu的id
            for gpu_id in range(num_gpu):
                #指定目标gpu
                with tf.device('/gpu:%d' % gpu_id):
                    print('tower:%d...' % gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):
                        #找到输入的起始和终止
                        start_pos = gpu_id * BATCH_SIZE
                        stop_pos = (gpu_id + 1) * BATCH_SIZE
                        #切分输入数据
                        x = images[start_pos:stop_pos]
                        y = labels[start_pos:stop_pos]

                        #得到每个模型的pred，loss，accuracy
                        pred, loss, acc = build_model(x, y, REG)
                        #设置variable为reuse
                        tf.get_variable_scope().reuse_variables()
                        #获取opt更新的当前tower的grads
                        grads = opt.compute_gradients(loss)
                        #打包给models数组
                        models.append((loss, grads, acc))

        print('build model on gpu tower done.')

        print('reduce model on cpu...')
        #通过zip(*models)来把同种数据放到一起
        #比如tower_losses是(tower1_loss,tower2_loss)
        tower_losses, tower_grads, tower_acc = zip(*models)
        #得到average的loss
        aver_loss_op = tf.reduce_mean(tower_losses)
        #得到更新gradients的op
        apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))
        #得到average的accuracy
        aver_acc_op = tf.reduce_mean(tower_acc)

        # 选择是否显示每个op和varible的物理位置
        config = tf.ConfigProto(log_device_placement=log_device_placement)
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
                    feed_dict[learning_rate] = lr
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
        # model数组为每个gpu的tuple数组
        models = []
        # 每个GPU模型的输入pl
        images = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.float32, [None, 10])
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:0'):
                # 得到pred, 和loss
                pred, loss, acc = build_model(images, labels, REG)

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
            for epoch in range(EPOCH_NUM):

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
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
log_device_placement = True

def average_losses(loss):
    #把loss加入到losses集合中
    tf.add_to_collection('losses', loss)

    # Assemble all of the losses for the current tower only.
    #得到losses集合
    losses = tf.get_collection('losses')

    # Calculate the total loss for the current tower.
    #得到正则化的loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #把loss和正则化的loss加到一起
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    return total_loss

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
                        pred, loss = build_model(x, y)

                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(loss)
                        models.append((x, y, pred, loss, grads))

        print('build model on gpu tower done.')

        print('reduce model on cpu...')
        tower_x, tower_y, tower_preds, tower_losses, tower_grads = zip(*models)
        aver_loss_op = tf.reduce_mean(tower_losses)
        apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))

        #?可以把这个部分分别放到各自GPU上运行
        all_y = tf.reshape(tf.stack(tower_y, 0), [-1, 10])
        all_pred = tf.reshape(tf.stack(tower_preds, 0), [-1, 10])
        correct_pred = tf.equal(tf.argmax(all_y, 1), tf.argmax(all_pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
        print('reduce model on cpu done.')

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
                print('\n---------------------')
                print('Epoch:%d, lr:%.4f' % (epoch, lr))
                for batch_idx in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    inp_dict = {}
                    inp_dict[learning_rate] = lr
                    inp_dict[images] = batch_x
                    inp_dict[labels] = batch_y
                    #inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y)
                    _, _loss = sess.run([apply_gradient_op, aver_loss_op], inp_dict)
                    avg_loss += _loss
                avg_loss /= total_batch

                lr = max(lr * 0.7, 0.00001)


                print('Train loss:%.4f' % (avg_loss))
                gpu_info = os.popen('nvidia-smi')
                print(gpu_info.read())

                ###########################################
                # VALIDATION PART
                ###########################################
                #val_payload_per_gpu = batch_size // num_gpu
                total_batch = int(mnist.validation.num_examples / batch_size)
                preds = None
                ys = None

                for batch_idx in range(total_batch):
                    batch_x, batch_y = mnist.validation.next_batch(batch_size)
                    #inp_dict = feed_all_gpu({}, models, val_payload_per_gpu, batch_x, batch_y)
                    inp_dict[images] = batch_x
                    inp_dict[labels] = batch_y
                    batch_pred, batch_y = sess.run([all_pred, all_y], inp_dict)
                    if preds is None:
                        preds = batch_pred
                    else:
                        preds = np.concatenate((preds, batch_pred), 0)
                    if ys is None:
                        ys = batch_y
                    else:
                        ys = np.concatenate((ys, batch_y), 0)
                val_accuracy = sess.run([accuracy], {all_y: ys, all_pred: preds})[0]
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
            preds = None
            ys = None
            for batch_idx in range(total_batch):
                batch_x, batch_y = mnist.test.next_batch(batch_size)
                #inp_dict = feed_all_gpu({}, models, test_payload_per_gpu, batch_x, batch_y)
                inp_dict[images] = batch_x
                inp_dict[labels] = batch_y
                batch_pred, batch_y = sess.run([all_pred, all_y], inp_dict)
                if preds is None:
                    preds = batch_pred
                else:
                    preds = np.concatenate((preds, batch_pred), 0)
                if ys is None:
                    ys = batch_y
                else:
                    ys = np.concatenate((ys, batch_y), 0)
            test_accuracy = sess.run([accuracy], {all_y: ys, all_pred: preds})[0]
            print('Test Accuracy: %0.4f%%\n\n' % (100.0 * test_accuracy))

if __name__ == '__main__':
    #single_gpu()
    multi_gpu(2)
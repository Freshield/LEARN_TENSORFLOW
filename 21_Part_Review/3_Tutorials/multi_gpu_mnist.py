import sys
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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

def _variable_on_cpu(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def get_weight_varible(name,shape):
    #return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return _variable_on_cpu(name, shape)

def get_bias_varible(name,shape):
    #return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return _variable_on_cpu(name, shape)
#filter_shape: [f_h, f_w, f_ic, f_oc]
def conv2d(layer_name, x, filter_shape):
    with tf.variable_scope(layer_name):
        w = get_weight_varible('w', filter_shape)
        b = get_bias_varible('b', filter_shape[-1])
        y = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME'), b)
        return y

def pool2d(layer_name, x):
    with tf.variable_scope(layer_name):
        y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return y

#inp_shape: [N, L]
#out_shape: [N, L]
def fc(layer_name, x, inp_shape, out_shape):
    with tf.variable_scope(layer_name):
        inp_dim = inp_shape[-1]
        out_dim = out_shape[-1]
        y = tf.reshape(x, shape=inp_shape)
        w = get_weight_varible('w', [inp_dim, out_dim])
        b = get_bias_varible('b', [out_dim])
        y = tf.add(tf.matmul(y, w), b)
        return y

def build_model(x):
    y = tf.reshape(x,shape=[-1, 28, 28, 1])
    #layer 1
    y = conv2d('conv_1', y, [3, 3, 1, 8])
    look_value('conv1',y)
    y = pool2d('pool_1', y)
    look_value('pool1',y)
    #layer 2
    y = conv2d('conv_2', y, [3, 3, 8, 16])
    look_value('conv2',y)
    y = pool2d('pool_2', y)
    look_value('pool2',y)
    #layer fc
    y = fc('fc', y, [-1, 7*7*16], [-1, 10])
    look_value('fc',y)
    return y

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

    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    """
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
        v = grad_and_vars[0][1]
        #这里的tuple是(平均的grads,variables)
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
        #最后averages相当于
        #[(avg_conv1.grads,conv1),(avg_bias1.grads,bias1),...]
    return average_grads

def feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y,):
    for i in range(len(models)):
        x, y, _, _, _ = models[i]
        start_pos = i * payload_per_gpu
        stop_pos = (i + 1) * payload_per_gpu
        inp_dict[x] = batch_x[start_pos:stop_pos]
        inp_dict[y] = batch_y[start_pos:stop_pos]
    return inp_dict

def multi_gpu(num_gpu):
    batch_size = BATCH_SIZE * num_gpu
    mnist = input_data.read_data_sets('/tmp/data/mnist',one_hot=True)

    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
        with tf.device('/cpu:0'):
            learning_rate = tf.placeholder(tf.float32, shape=[])
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

            print('build model...')
            print('build model on gpu tower...')
            models = []
            for gpu_id in range(num_gpu):
                with tf.device('/gpu:%d' % gpu_id):
                    print('tower:%d...'% gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope('cpu_variables', reuse=gpu_id>0):
                            x = tf.placeholder(tf.float32, [None, 784])
                            y = tf.placeholder(tf.float32, [None, 10])
                            pred = build_model(x)
                            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
                            grads = opt.compute_gradients(loss)
                            models.append((x,y,pred,loss,grads))
            print('build model on gpu tower done.')

            print('reduce model on cpu...')
            tower_x, tower_y, tower_preds, tower_losses, tower_grads = zip(*models)
            aver_loss_op = tf.reduce_mean(tower_losses)
            apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))

            all_y = tf.reshape(tf.stack(tower_y, 0), [-1,10])
            all_pred = tf.reshape(tf.stack(tower_preds, 0), [-1,10])
            correct_pred = tf.equal(tf.argmax(all_y, 1), tf.argmax(all_pred, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
            print('reduce model on cpu done.')

            print('run train op...')
            sess.run(tf.global_variables_initializer())
            lr = 0.01
            for epoch in range(10):
                start_time = time.time()
                payload_per_gpu = batch_size // num_gpu
                total_batch = int(mnist.train.num_examples/batch_size)
                avg_loss = 0.0
                print('\n---------------------')
                print('Epoch:%d, lr:%.4f' % (epoch,lr))
                for batch_idx in range(total_batch):
                    batch_x,batch_y = mnist.train.next_batch(batch_size)
                    inp_dict = {}
                    inp_dict[learning_rate] = lr
                    inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y)
                    _, _loss = sess.run([apply_gradient_op, aver_loss_op], inp_dict)
                    avg_loss += _loss
                avg_loss /= total_batch
                print('Train loss:%.4f' % (avg_loss))

                lr = max(lr * 0.7,0.00001)

                val_payload_per_gpu = batch_size // num_gpu
                total_batch = int(mnist.validation.num_examples / batch_size)
                preds = None
                ys = None
                for batch_idx in range(total_batch):
                    batch_x,batch_y = mnist.validation.next_batch(batch_size)
                    inp_dict = feed_all_gpu({}, models, val_payload_per_gpu, batch_x, batch_y)
                    batch_pred,batch_y = sess.run([all_pred,all_y], inp_dict)
                    if preds is None:
                        preds = batch_pred
                    else:
                        preds = np.concatenate((preds, batch_pred), 0)
                    if ys is None:
                        ys = batch_y
                    else:
                        ys = np.concatenate((ys,batch_y),0)
                val_accuracy = sess.run([accuracy], {all_y:ys, all_pred:preds})[0]
                print('Val Accuracy: %0.4f%%' % (100.0 * val_accuracy))

                stop_time = time.time()
                elapsed_time = stop_time-start_time
                print('Cost time: ' + str(elapsed_time) + ' sec.')
            print('training done.')

            test_payload_per_gpu = batch_size // num_gpu
            total_batch = int(mnist.test.num_examples / batch_size)
            preds = None
            ys = None
            for batch_idx in range(total_batch):
                batch_x, batch_y = mnist.test.next_batch(batch_size)
                inp_dict = feed_all_gpu({}, models, test_payload_per_gpu, batch_x, batch_y)
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
    multi_gpu(2)
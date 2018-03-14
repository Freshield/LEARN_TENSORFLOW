import tensorflow as tf
from tensorflow.python.training import moving_averages


epsilon = 1e-5

#需要用get_variable来共享variable并把权重放到cpu
def _variable_on_cpu(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def get_weight_varible(name,shape, initializer=tf.contrib.layers.xavier_initializer()):
    #return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return _variable_on_cpu(name, shape, initializer)

def get_bias_varible(name,shape, initializer=tf.contrib.layers.xavier_initializer()):
    #return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return _variable_on_cpu(name, shape, initializer)
#filter_shape: [f_h, f_w, f_ic, f_oc]
def conv2d(layer_name, x, filter_shape):
    with tf.variable_scope(layer_name):
        para = []

        w = get_weight_varible('w', filter_shape)
        b = get_bias_varible('b', filter_shape[-1])
        y = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME'), b)

        para.append(w)
        para.append(b)

        return y, para

def pool2d(layer_name, x):
    with tf.variable_scope(layer_name):
        y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return y

#inp_shape: [N, L]
#out_shape: [N, L]
def fc(layer_name, x, inp_shape, out_shape):
    with tf.variable_scope(layer_name):
        para = []

        inp_dim = inp_shape[-1]
        out_dim = out_shape[-1]
        y = tf.reshape(x, shape=inp_shape)
        w = get_weight_varible('w', [inp_dim, out_dim])
        b = get_bias_varible('b', [out_dim])
        y = tf.add(tf.matmul(y, w), b)

        para.append(w)
        para.append(b)

        return y, para

def batch_norm(x, phase_train=True, scope='bn_conv'):

  with tf.variable_scope(scope):
      n_out = x.shape[-1]
      beta = tf.get_variable('beta_conv', shape=[n_out], initializer=tf.constant_initializer(0.0))
      gamma = tf.get_variable('gamma_conv', shape=[n_out], initializer=tf.constant_initializer(1.0))

      batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')

      moving_mean = tf.get_variable('batch_mean', shape=batch_mean.get_shape(), initializer=tf.constant_initializer(0.0), trainable=False)
      moving_variance = tf.get_variable('batch_var', shape=batch_var.get_shape(), initializer=tf.constant_initializer(0.0), trainable=False)

      update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                 batch_mean, 0.5, zero_debias=False)
      update_moving_variance = moving_averages.assign_moving_average(
                                                          moving_variance, batch_var, 0.5, zero_debias=False)
      def mean_var_with_update():
          with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                  return tf.identity(batch_mean), tf.identity(batch_var)

      mean, var = tf.cond(phase_train,
                                        mean_var_with_update,
                                        lambda: (moving_mean, moving_variance))

      normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
  return normed

def build_model(x, y, reg, phase_train):
    para = []

    input_layer = tf.reshape(x,shape=[-1, 28, 28, 1])
    #layer 1
    conv1_layer, conv1_para = conv2d('conv_1', input_layer, [3, 3, 1, 8])
    conv1_bn = batch_norm(conv1_layer,phase_train)
    para += conv1_para

    pool1_layer = pool2d('pool_1', conv1_bn)

    #layer 2
    conv2_layer, conv2_para = conv2d('conv_2', pool1_layer, [3, 3, 8, 16])
    para += conv2_para

    pool2_layer = pool2d('pool_2', conv2_layer)
    #layer fc
    pred, fc_para = fc('fc', pool2_layer, [-1, 7*7*16], [-1, 10])
    para += fc_para

    reg_loss = tf.zeros([])
    for p in para:
        reg_loss += reg * 0.5 * tf.nn.l2_loss(p)
    loss = reg_loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    return loss, accuracy

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

      Note that this function provides a synchronization point across all towers.

      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
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
        print(grad_and_vars)
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

def tower_model(images, labels, PARA, phase_train):
    # 获取optimizer
    opt = tf.train.AdamOptimizer(learning_rate=PARA.LEARNING_RATE)

    print('build model...')
    print('build model on gpu tower...')
    # model数组为每个gpu的tuple数组
    models = []
    # 这里来做不同的GPU的tower模型
    with tf.variable_scope(tf.get_variable_scope()):
        # 获取gpu的id
        for gpu_id in range(PARA.NUM_GPU):
            # 指定目标gpu
            with tf.device('/gpu:%d' % gpu_id):
                print('tower:%d...' % gpu_id)
                with tf.name_scope('tower_%d' % gpu_id):
                    # 找到输入的起始和终止
                    start_pos = gpu_id * PARA.BATCH_SIZE
                    stop_pos = (gpu_id + 1) * PARA.BATCH_SIZE
                    # 切分输入数据
                    x = images[start_pos:stop_pos]
                    y = labels[start_pos:stop_pos]
                    # 得到每个模型的loss，accuracy
                    loss, acc = build_model(x, y, PARA.REG, phase_train)
                    # 设置variable为reuse
                    tf.get_variable_scope().reuse_variables()
                    # 获取opt更新的当前tower的grads
                    grads = opt.compute_gradients(loss)
                    # 打包给models数组
                    models.append((loss, grads, acc))

    print('build model on gpu tower done.')

    print('reduce model on cpu...')
    # 通过zip(*models)来把同种数据放到一起
    # 比如tower_losses是(tower1_loss,tower2_loss)
    tower_losses, tower_grads, tower_acc = zip(*models)
    # 得到average的loss
    aver_loss_op = tf.reduce_mean(tower_losses)
    # 得到更新gradients的op
    apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))
    # 得到average的accuracy
    aver_acc_op = tf.reduce_mean(tower_acc)

    return apply_gradient_op, aver_loss_op, aver_acc_op

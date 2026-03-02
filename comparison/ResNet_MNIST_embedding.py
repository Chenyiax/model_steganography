"""
这个代码并不是徐辰屹写的
所以没什么注释
方法是在训练过程中嵌入秘密信息
如果将超参数 α 设置为 0,就是 Multi-source Data Hiding in Neural Networks 的方法
如果 α 设置为 2,就是 A general steganographic framework for neural network models 的方法
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 指定第五块GPU跑

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

capacity = 5000  # 隐写容量
beta = 200       # 超参数β
alpha = 2        # 超参数α


def weight_variable(shape):
    # 这里是构建初始变量
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    # 创建变量
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 在这里定义残差网络的id_block块，此时输入和输出维度相同
def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block):
    """
        Implementation of the identity block as defined in Figure 3

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

    # defining name basis
    block_name = 'res' + str(stage) + block
    f1, f2, f3 = out_filters
    with tf.variable_scope(block_name):
        X_shortcut = X_input

        # first
        W_conv1 = weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        b_conv1 = bias_variable([f1])
        X = tf.nn.relu(X + b_conv1)

        # second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        b_conv2 = bias_variable([f2])
        X = tf.nn.relu(X + b_conv2)

        # third

        W_conv3 = weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        b_conv3 = bias_variable([f3])
        X = tf.nn.relu(X + b_conv3)
        # final step
        add = tf.add(X, X_shortcut)
        # b_conv_fin = bias_variable([f3])
        add_result = tf.nn.relu(add)

    return add_result


# 在这里定义残差网络的id_block块，此时输入和输出维度相同
def identity_block_embed(X_input, kernel_size, in_filter, out_filters, stage, block):
    """
        Implementation of the identity block as defined in Figure 3

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

    # defining name basis
    block_name = 'res' + str(stage) + block
    f1, f2, f3 = out_filters
    with tf.variable_scope(block_name):
        X_shortcut = X_input

        # first
        W_conv1 = weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        b_conv1 = bias_variable([f1])
        X = tf.nn.relu(X + b_conv1)

        # second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        b_conv2 = bias_variable([f2])

        c_data_1 = tf.reduce_mean(W_conv2, 0)
        c_data = tf.reduce_mean(c_data_1, 0)
        c_data = tf.reshape(c_data, (1, f1 * f2))
        x_random = np.random.randint(2, size=(f1 * f2, capacity))

        X = tf.nn.relu(X + b_conv2)

        # third

        W_conv3 = weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        b_conv3 = bias_variable([f3])
        X = tf.nn.relu(X + b_conv3)
        # final step
        add = tf.add(X, X_shortcut)
        # b_conv_fin = bias_variable([f3])
        add_result = tf.nn.relu(add)

    return add_result, c_data, x_random


# 这里定义conv_block模块，由于该模块定义时输入和输出尺度不同，故需要进行卷积操作来改变尺度，从而得以相加
def convolutional_block_embed(X_input, kernel_size, in_filter,
                              out_filters, stage, block, stride=2):
    """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

    # defining name basis
    block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters

        x_shortcut = X_input
        # first
        W_conv1 = weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='SAME')
        b_conv1 = bias_variable([f1])
        X = tf.nn.relu(X + b_conv1)

        # second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        """
            在W_conv2嵌入操作
            """

        c_data_1 = tf.reduce_mean(W_conv2, 0)
        c_data = tf.reduce_mean(c_data_1, 0)
        c_data = tf.reshape(c_data, (1, f1 * f2))
        x_random = np.random.randint(2, size=(f1 * f2, capacity))
        w_conv = tf.layers.flatten(W_conv2)
        # w_conv = tf.layers.flatten(w_conv)

        b_conv2 = bias_variable([f2])
        X = tf.nn.relu(X + b_conv2)

        # third
        W_conv3 = weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        b_conv3 = bias_variable([f3])
        X = tf.nn.relu(X + b_conv3)
        # shortcut path
        W_shortcut = weight_variable([1, 1, in_filter, f3])
        x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

        # final
        add = tf.add(x_shortcut, X)
        # 建立最后融合的权重
        # b_conv_fin = bias_variable([f3])
        add_result = tf.nn.relu(add)

    return add_result, c_data, x_random.astype(np.float32), w_conv


# 这里定义conv_block模块，由于该模块定义时输入和输出尺度不同，故需要进行卷积操作来改变尺度，从而得以相加
def convolutional_block(X_input, kernel_size, in_filter,
                        out_filters, stage, block, stride=2):
    """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

    # defining name basis
    block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters

        x_shortcut = X_input
        # first
        W_conv1 = weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='SAME')
        b_conv1 = bias_variable([f1])
        X = tf.nn.relu(X + b_conv1)

        # second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        b_conv2 = bias_variable([f2])
        X = tf.nn.relu(X + b_conv2)

        # third
        W_conv3 = weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        b_conv3 = bias_variable([f3])
        X = tf.nn.relu(X + b_conv3)
        # shortcut path
        W_shortcut = weight_variable([1, 1, in_filter, f3])
        x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

        # final
        add = tf.add(x_shortcut, X)
        # 建立最后融合的权重
        # b_conv_fin = bias_variable([f3])
        add_result = tf.nn.relu(add)

    return add_result


if __name__ == "__main__":

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    z = tf.placeholder(shape=[None, capacity], dtype=tf.float32, name="z")

    sess = tf.InteractiveSession()

    x1 = tf.reshape(x, [-1, 28, 28, 1])
    w_conv1 = weight_variable([2, 2, 1, 64])
    x1 = tf.nn.conv2d(x1, w_conv1, strides=[1, 2, 2, 1], padding='SAME')
    b_conv1 = bias_variable([64])
    x1 = tf.nn.relu(x1 + b_conv1)
    # 这里操作后变成14x14x64
    x1 = tf.nn.max_pool(x1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    # stage 2
    x2, c_data, x_random, w_conv = convolutional_block_embed(X_input=x1, kernel_size=3, in_filter=64,
                                                             out_filters=[64, 64, 256],
                                                             stage=2, block='a',
                                                             stride=1)

    # 上述conv_block操作后，尺寸变为14x14x256
    x2 = identity_block(x2, 3, 256, [64, 64, 256], stage=2, block='b')
    x2 = identity_block(x2, 3, 256, [64, 64, 256], stage=2, block='c')
    # 上述操作后张量尺寸变成14x14x256
    x2 = tf.nn.max_pool(x2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 变成7x7x256
    flat = tf.reshape(x2, [-1, 7 * 7 * 256])
    decoder_data_output = tf.matmul(c_data, x_random)
    decoder_data = tf.nn.sigmoid(decoder_data_output)
    decoder_data_round = tf.round(decoder_data)

    w_fc1 = weight_variable([7 * 7 * 256, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    # 建立损失函数，在这里采用交叉熵函数
    loss_data_mse = tf.losses.mean_squared_error(z, decoder_data)
    loss_data_round_mse = tf.losses.mean_squared_error(z, decoder_data_round)
    loss_data = beta * loss_data_mse + alpha * loss_data_round_mse

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    loss = tf.add(loss_data, cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化变量

    sess.run(tf.global_variables_initializer())

    batch_size = 50
    # capacity = b
    watermarking_sub = np.rint(np.random.rand(1, capacity))
    watermarking = watermarking_sub.copy()
    for i in range(batch_size - 1):
        watermarking = np.row_stack((watermarking, watermarking_sub))

    extraction_error = 0.5

    saver = tf.train.Saver()

    # 恢复模型
    saver.restore(sess, './resnet_initial_weights.ckpt')

    for i in range(10000):
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict=
                                           {x: batch[0],
                                            y: batch[1],
                                            z: watermarking,
                                            keep_prob: 1.0})
            print("\nstep: {:.1f} training accuracy: {:.4f} extraction error: {:.4f}".format(i, train_accuracy,
                                                                                             extraction_error), end="")
        train_step.run(
            feed_dict={x: batch[0],
                       y: batch[1],
                       z: watermarking,
                       keep_prob: 0.5})
        data_val = sess.run([decoder_data],
                            feed_dict={x: mnist.test.images,
                                       y: mnist.test.labels,
                                       keep_prob: 1.0})
        # extracted_data = data_val.copy() + np.random.normal(0,0.08,np.shape(data_val))
        extracted_data = data_val.copy() + np.random.normal(0, 0.08, np.shape(data_val))
        extracted_data = np.rint(extracted_data)
        extraction_error = np.sum(np.abs(watermarking - extracted_data)) / (batch_size * capacity)

    accuracy_test = accuracy.eval(
        feed_dict={x: mnist.test.images,
                   y: mnist.test.labels,
                   z: watermarking,
                   keep_prob: 1.0})
    wconv, loss = sess.run([w_conv, cross_entropy],
                           feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels,
                                      z: watermarking,
                                      keep_prob: 1.0})


    print("\ntest accuracy: {:.4f} extraction error: {:.4f}".format(accuracy_test, extraction_error), end="")
    np.save('resnet_weight_with_secret23.npy', wconv)
    sess.close()
    # time_end = time.time()

    # wconv = tf.layers.flatten(wconv)
    # wconv = np.array(wconv)
    # x_conv_res_mnist = np.transpose(wconv[0, :])
    # np.save('./results_pic/res_embed_mnist.npy', x_conv_res_mnist)
    # plt.hist(x_conv_res_mnist, bins=300, range=(-1, 1), color='b', edgecolor='grey')
    # plt.xlabel('Parameter values')
    # plt.ylabel('Numbers')
    # plt.show()

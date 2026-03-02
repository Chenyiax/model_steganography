from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# c_data=4096


capacity = 5000
beta = 5.1
# alpha = 0
alpha = beta/2
z = tf.placeholder(shape=[None, capacity], dtype=tf.float32, name="z")


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_features])
    return conv


def conv2d_embed(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')

    c_data_1 = tf.reduce_mean(W, 0)
    c_data = tf.reduce_mean(c_data_1, 0)
    c_data = tf.reshape(c_data, (1, in_features * out_features))
    x_random = np.random.randint(2, size=(in_features * out_features, capacity))

    w_conv = tf.layers.flatten(W)
    if with_bias:
        return conv + bias_variable([out_features])
    return conv, c_data, x_random, w_conv


def avg_pool(input, s):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, s, s, 1], 'SAME')


def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
    # current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.layers.batch_normalization(current, momentum=0.9, center=True, scale=True, epsilon=1e-3, training=True)
    current = tf.nn.relu(current)
    current = conv2d(current, in_features, out_features, kernel_size)
    current = tf.nn.dropout(current, keep_prob)
    return current


def batch_activ_conv_embed(current, in_features, out_features, kernel_size, is_training, keep_prob):
    # current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.layers.batch_normalization(current, momentum=0.9, center=True, scale=True, epsilon=1e-3, training=True)
    current = tf.nn.relu(current)
    current, c_data, x_random, w_conv = conv2d_embed(current, in_features, out_features, kernel_size)
    current = tf.nn.dropout(current, keep_prob)
    return current, c_data, x_random, w_conv


# 定义稠密神经网络的稠密块
def block(input, layers, in_features, growth, is_training, keep_prob):
    current = input
    features = in_features
    for idx in range(layers):
        tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
        current = tf.concat((current, tmp), axis=3)
        features += growth
    return current, features


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

if __name__ == '__main__':

    y = tf.placeholder(tf.float32, [None, 10])
    inputs_ = tf.placeholder(tf.float32, [None, 784])

    keep_prob = tf.placeholder(tf.float32)
    # is_training = tf.placeholder("bool", shape=[])
    is_training = True
    sess = tf.InteractiveSession()

    x = tf.reshape(inputs_, (-1, 28, 28, 1))
    conv1 = conv2d(x, 1, 16, 3)
    # 这里操作后变成28*28*16
    current, features = block(conv1, layers=4, in_features=16, growth=12, is_training=is_training, keep_prob=keep_prob)
    current, c_data, x_random, w_conv = batch_activ_conv_embed(current, features, features, 1, is_training=is_training,
                                                               keep_prob=keep_prob)
    x_random = tf.cast(x_random, tf.float32)
    decoder_data_output = tf.matmul(c_data, x_random)
    decoder_data = tf.nn.sigmoid(decoder_data_output)
    decoder_data_round = tf.round(decoder_data)

    current = avg_pool(current, 2)
    current, features = block(current, layers=4, in_features=features, growth=12, is_training=is_training,
                              keep_prob=keep_prob)
    current = batch_activ_conv(current, features, features, 1, is_training=is_training, keep_prob=keep_prob)
    current = avg_pool(current, 2)

    current, features = block(current, layers=4, in_features=features, growth=12, is_training=is_training,
                              keep_prob=keep_prob)
    # current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.layers.batch_normalization(current, momentum=0.9, center=True, scale=True, epsilon=1e-3, training=True)

    current = tf.nn.relu(current)
    # current = avg_pool(current, 8)
    current = avg_pool(current, 2)
    final_dim = features
    label_count = 1024
    flat = tf.reshape(current, [-1, 4 * 4 * final_dim])
    Wfc = weight_variable([4 * 4 * final_dim, label_count])
    bfc = bias_variable([label_count])

    h_fc1 = tf.nn.relu(tf.matmul(flat, Wfc) + bfc)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    ys_ = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    # 建立损失函数，在这里采用交叉熵函数

    loss_data_l2 = tf.norm(decoder_data_output, 2)
    loss_data_mse_round = tf.losses.mean_squared_error(z, decoder_data_round)
    loss_data_mse = tf.losses.mean_squared_error(z, decoder_data)

    loss_data = tf.add(beta * loss_data_mse, alpha * loss_data_mse_round)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=ys_))
    loss = tf.add(loss_data, cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(y, 1))
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

    print("start")
    for i in range(10000):
        batch = mnist.train.next_batch(batch_size)
        # input = batch[0].reshape((-1, 28, 28, 1))
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict=
                                           {inputs_: batch[0],
                                            y: batch[1],
                                            z: watermarking,
                                            keep_prob: 1.0})
            print("\nstep: {:.1f} training accuracy: {:.4f} extraction error: {:.4f}".format(i, train_accuracy,
                                                                                             extraction_error), end="")
        train_step.run(
            feed_dict={inputs_: batch[0],
                       y: batch[1],
                       z: watermarking,
                       keep_prob: 0.5})
        data_val = sess.run([decoder_data],
                            feed_dict={inputs_: mnist.test.images,
                                       y: mnist.test.labels,
                                       keep_prob: 1.0})
        extracted_data = data_val.copy()
        extracted_data = np.rint(extracted_data)
        extraction_error = np.sum(np.abs(watermarking - extracted_data)) / (batch_size * capacity)
        # train_step.run(feed_dict={x: input, y: batch[1], keep_prob: 1.0})
    accuracy_test = accuracy.eval(
        feed_dict={inputs_: mnist.test.images,
                   y: mnist.test.labels,
                   z: watermarking,
                   keep_prob: 1.0})

    wconv = sess.run([w_conv], feed_dict={inputs_: mnist.test.images,
                                          y: mnist.test.labels,
                                          z: watermarking,
                                          keep_prob: 1.0})
    print("\ntest accuracy: {:.4f} extraction error: {:.4f}".format(accuracy_test, extraction_error), end="")
    wconv = np.array(wconv)
    np.save('densenet_weight_with_secret23.npy', wconv)
    sess.close()
    #
    # wconv = np.array(wconv)
    # x_conv = np.transpose(wconv[0, :])
    # np.save('./results_pic/dense_embed_conv.npy', x_conv)
    # plt.hist(x_conv, bins=30,  color='b', edgecolor='b')
    # plt.xlabel('Parameter values')
    # plt.ylabel('Numbers')
    # plt.show()

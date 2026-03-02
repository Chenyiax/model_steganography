# coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# capacity = 5000#嵌入容量
# beta = 4000#β
# alpha = 0

# paras
n_classes = 10
# Training Parameters
learning_rate = 0.001
num_steps = 10000
batch_size = 50
display_step = 10

X = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, n_classes])
# z1 = tf.placeholder(tf.float32, [None, capacity])

keep_prob = tf.placeholder(tf.float32)


# build vgg16 model
x = tf.reshape(X, [-1, 28, 28, 1])
tf.summary.image('x', x)
# conv_1
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 16], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(out, name='scope')
with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 16, 16], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(out, name='scope')
# pool1
pool_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')

# conv_2
with tf.name_scope('conv2_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(out, name='scope')
with tf.name_scope('conv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 32], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(out, name='scope')

# pool2
pool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')

# conv_3
with tf.name_scope('conv3_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(out, name='scope')
with tf.name_scope('conv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(out, name='scope')
with tf.name_scope('conv3_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_3 = tf.nn.relu(out, name='scope')
# pool_3
pool_3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_3')

# conv_4
with tf.name_scope('conv4_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights')
    w_conv = tf.layers.flatten(kernel)
    conv = tf.nn.conv2d(pool_3, kernel, [1, 1, 1, 1], padding='SAME')

    # c_data_1 = tf.reduce_mean(kernel, 0)
    # c_data = tf.reduce_mean(c_data_1, 0)
    # c_data = tf.reshape(c_data, (1, 8192))
    # x_random1 = np.random.randint(2, size=(8192, capacity))
    #
    # decoder_data_output1 = tf.matmul(c_data, x_random1)
    # decoder_data1 = tf.nn.sigmoid(decoder_data_output1)
    # decoder_data_round1 = tf.round(decoder_data1)

    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(out, name='scope')
with tf.name_scope('conv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(out, name='scope')
with tf.name_scope('conv4_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_3 = tf.nn.relu(out, name='scope')
# pool_4
pool_4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_4')

# conv_5
with tf.name_scope('conv5_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool_4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_1 = tf.nn.relu(out, name='scope')
with tf.name_scope('conv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_2 = tf.nn.relu(out, name='scope')
with tf.name_scope('conv5_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_3 = tf.nn.relu(out, name='scope')
# pool_5
pool_5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_5')

# fc1
with tf.name_scope('fc1') as scope:
    shape = int(np.prod(pool_5.get_shape()[1:]))
    fc1w = tf.Variable(tf.truncated_normal([shape, 100], dtype=tf.float32, stddev=1e-1), name='weights')
    fc1b = tf.Variable(tf.constant(1.0, shape=[100], dtype=tf.float32), trainable=True, name='biases')
    pool5_flat = tf.reshape(pool_5, [-1, shape])
    fc11 = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
    fc1 = tf.nn.relu(fc11, name='scope')
# fc2
with tf.name_scope('fc2') as scope:
    fc2w = tf.Variable(tf.truncated_normal([100, 100], dtype=tf.float32, stddev=1e-1), name='weights')
    fc2b = tf.Variable(tf.constant(1.0, shape=[100], dtype=tf.float32), trainable=True, name='biases')
    fc21 = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
    fc2 = tf.nn.relu(fc21, name='scope')
# fc3
with tf.name_scope('fc3') as scope:
    fc3w = tf.Variable(tf.truncated_normal([100, 10], dtype=tf.float32, stddev=1e-1), name='weights')
    fc3b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32), trainable=True, name='biases')
    fc31 = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b, name='scope')

mnist = input_data.read_data_sets('../data/', one_hot=True)

prediction = tf.nn.softmax(fc31)

# tensorboard事件保存地址
log_dir = './tensorboard/'
# Define loss and optimizer
with tf.name_scope('loss'):
    # loss_data = tf.losses.mean_squared_error(z, decoder_data)
    # loss_data_mse1 = tf.losses.mean_squared_error(z1, decoder_data1)
    # loss_data_mse_round1 = tf.losses.mean_squared_error(z1, decoder_data_round1)
    # loss_data1 = tf.add(beta * loss_data_mse1, alpha*loss_data_mse_round1)


    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc31, labels=y))
    # loss = tf.add(loss_data1, loss_op)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

# evaluate model
with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# tensorboard
tf.summary.scalar('loss', loss_op)
tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
# initialize the variables
init = tf.global_variables_initializer()

# watermarking_sub1 = np.rint(np.random.rand(1, capacity))
# watermarking1 = watermarking_sub1.copy()
# for i in range(batch_size - 1):
#     watermarking1 = np.row_stack((watermarking1, watermarking_sub1))


# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # extraction_error = 0.5

    for step in range(num_steps):
        batch = mnist.train.next_batch(100)
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        if step % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict=
                                           {X: batch[0],
                                            y: batch[1],
                                            # z1:watermarking1,
                                            keep_prob: 1.0})
            # print('step %d, training accuracy %g' % (step, train_accuracy))
            print("\nstep: {:.1f} training accuracy: {:.4f}".format(step, train_accuracy), end="")
            train_loss = loss_op.eval(feed_dict=
                                           {X: batch[0],
                                            y: batch[1],
                                            # z1:watermarking1,
                                            keep_prob: 1.0})
            # print('step %d, training accuracy %g' % (step, train_accuracy))
            print("\nstep: {:.1f} training loss: {:.4f}".format(step,train_loss), end="")

            train_lossop = loss_op.eval(feed_dict=
                                           {X: batch[0],
                                            y: batch[1],
                                            # z1:watermarking1,
                                            keep_prob: 1.0})
            # print('step %d, training accuracy %g' % (step, train_accuracy))
            print("\nstep: {:.1f} training loss_op: {:.4f}".format(step,train_lossop), end="")

        train_op.run(
            feed_dict={X: batch[0],
                       y: batch[1],
                       # z1:watermarking1,
                       keep_prob: 0.5})
        # data_val1 = sess.run([decoder_data1],
        #                     feed_dict={X: mnist.test.images,
        #                                y: mnist.test.labels,
        #                                z1: watermarking1,
        #                                keep_prob: 1.0})
        # extracted_data1 = data_val1.copy()
        # extracted_data1 = np.rint(extracted_data1)
        # extraction_error = np.sum(np.abs(watermarking1 - extracted_data1)) / (batch_size * capacity)

    accuracy_test = accuracy.eval(
        feed_dict={X: mnist.test.images,
                   y: mnist.test.labels,
                   # z1: watermarking1,
                   keep_prob: 1.0})
    wconv, loss_op = sess.run([w_conv, loss_op], feed_dict={
        X: mnist.test.images,
        y: mnist.test.labels,
        keep_prob: 1.0})

    print("\ntest accuracy: {:.4f}".format(accuracy_test), end="")
    sess.close()


wconv = np.array(wconv)
x_conv = np.transpose(wconv[0, :])
np.save('vgg_weight_without_secret.npy', wconv)

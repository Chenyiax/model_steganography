#!/usr/bin/env python
# coding: utf-8

# In[1]:

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

# mnist = input_data.read_data_sets("/temp/data", one_hot = True)#加载MINST数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import time
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# In[2]:


sess = tf.InteractiveSession()


# In[3]:

 
b = 2000
beta = 1200
alpha = 0
z = tf.placeholder(shape=[None,b], dtype=tf.float32, name="z")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

def norm(x, lsize = 4):
    return tf.nn.lrn(x, lsize, bias = 1.0, alpha = 0.001/9.0, beta = 0.75)


# In[4]:


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1,28,28,1])


# In[5]:


W_conv1 = weight_variable([3,3,1,64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_3x3(h_conv1)
h_norm1 = norm(h_pool1, lsize = 4)


# In[6]:


W_conv2 = weight_variable([3,3,64,128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2) + b_conv2)
h_pool2 = max_pool_3x3(h_conv2)
h_norm2 = norm(h_pool2, lsize = 4)


# In[7]:


W_conv3 = weight_variable([3,3,128,256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_norm2, W_conv3) + b_conv3)
h_pool3 = max_pool_3x3(h_conv3)
h_norm3 = norm(h_pool3, lsize = 4)


# In[8]:


W_fc1 = weight_variable([4*4*256, 1024])
b_fc1 = bias_variable([1024])
h_norm3_flat = tf.reshape(h_norm3, [-1, 4*4*256])
h_fc1 = tf.nn.relu(tf.matmul(h_norm3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[9]:


W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


# In[10]:


W_fc3 = weight_variable([1024, 10])
b_fc3 = bias_variable([10])
y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3


# In[11]:


fc1 = tf.layers.flatten(W_fc1)
fc2 = tf.layers.flatten(W_fc2)
fc3 = tf.layers.flatten(W_fc3)
conv1 = tf.layers.flatten(W_conv1)
conv2 = tf.layers.flatten(W_conv2)
conv3 = tf.layers.flatten(W_conv3)

c_data_1 = tf.reduce_mean(W_conv2,0)
c_data = tf.reduce_mean(c_data_1,0)
c_data = tf.reshape(c_data,(1,8192))


# In[13]:


x_random = np.random.randint(2,size=(8192,b))
x_random = tf.cast(x_random, tf.float32)
decoder_data_output = tf.matmul(c_data,x_random)
decoder_data = tf.nn.sigmoid(decoder_data_output)
decoder_data_round = tf.round(decoder_data)

# In[14]:

loss_data_l2 = tf.norm(decoder_data_output, 2)
loss_data_mse_round = tf.losses.mean_squared_error(z, decoder_data_round)
loss_data_mse = tf.losses.mean_squared_error(z, decoder_data)


loss_data = tf.add(beta * loss_data_mse , alpha * loss_data_mse_round)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_conv))
loss = tf.add(loss_data, cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[15]:


batch_size = 50
capacity = b
watermarking_sub = np.rint(np.random.rand(1,capacity))
watermarking = watermarking_sub.copy()
for i in range(batch_size-1):
    watermarking = np.row_stack((watermarking,watermarking_sub))


# In[16]:


time_start = time.time()
sess.run(tf.global_variables_initializer())
extraction_error = 0.5
for i in range(10000):
    batch = mnist.train.next_batch(batch_size)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(
                feed_dict = {x: batch[0],
                             y: batch[1],
                             z: watermarking,
                             keep_prob: 1.0})
        print("\nstep: {:.1f} training accuracy: {:.4f} extraction error: {:.4f}".format(i, train_accuracy,
              extraction_error), end="")
    train_step.run(
            feed_dict = {x: batch[0],
                         y: batch[1],
                         z: watermarking,
                         keep_prob: 0.5})
    data_val = sess.run([decoder_data],
                        feed_dict = {x: mnist.test.images,
                                     y: mnist.test.labels,
                                     z: watermarking,
                                     keep_prob: 1.0})
    extracted_data = data_val.copy()
    extracted_data = np.rint(extracted_data)
    extraction_error = np.sum(np.abs(watermarking-extracted_data))/(batch_size*capacity)
    
accuracy_test = accuracy.eval(
        feed_dict = {x: mnist.test.images,
                     y: mnist.test.labels,
                     z: watermarking,
                     keep_prob: 1.0})
w_conv2 = sess.run([conv2], feed_dict = {x: mnist.test.images, y: mnist.test.labels, z: watermarking, keep_prob: 1.0})
# w_fc1, w_fc2, w_fc3, w_conv1, w_conv2, w_conv3 = sess.run([fc1, fc2, fc3, conv1, conv2, conv3], feed_dict = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})

print("\ntest accuracy: {:.4f} extraction error: {:.4f}".format(accuracy_test, extraction_error), end="")
sess.close()
time_end = time.time()
w_conv2 = np.array(w_conv2)
np.save('alexnet_weight_with_secret22.npy', w_conv2)
print('\ntime cost: ', time_end - time_start)


# In[17]:

#
# x = np.transpose(w_conv2[0, :])
# np.save('./W_conv2_em.npy', x)
# plt.hist(x, bins=50, range=(-0.5, 0.5), color='b', edgecolor='grey')
# plt.xlabel('Parameter values')
# plt.ylabel('Numbers')
# plt.show()
#
#
# # In[18]:
#
#
# x = np.transpose(w_conv3[0, :])
# np.save('C:/Watermarking in CapsNet/AlexNet/W_conv3_em.npy', x)
# plt.hist(x, bins=50, range=(-0.5, 0.5), color='b', edgecolor='grey')
# plt.xlabel('Parameter values')
# plt.ylabel('Numbers')
# plt.show()
#
#
# # In[19]:
#
#
# x = np.transpose(w_conv1[0, :])
# np.save('C:/Watermarking in CapsNet/AlexNet/W_conv1_em.npy', x)
# plt.hist(x, bins=50, range=(-0.5, 0.5), color='b', edgecolor='grey')
# plt.xlabel('Parameter values')
# plt.ylabel('Numbers')
# plt.show()
#
#
# # In[20]:
#
#
# x = np.transpose(w_fc1[0, :])
# np.save('C:/Watermarking in CapsNet/AlexNet/W_fc1_em.npy', x)
# plt.hist(x, bins=50, range=(-0.5, 0.5), color='b', edgecolor='grey')
# plt.xlabel('Parameter values')
# plt.ylabel('Numbers')
# plt.show()
#
#
# # In[21]:
#
#
# x = np.transpose(w_fc2[0, :])
# np.save('C:/Watermarking in CapsNet/AlexNet/W_fc2_em.npy', x)
# plt.hist(x, bins=50, range=(-0.5, 0.5), color='b', edgecolor='grey')
# plt.xlabel('Parameter values')
# plt.ylabel('Numbers')
# plt.show()
#
#
# # In[22]:
#
#
# x = np.transpose(w_fc3[0, :])
# np.save('C:/Watermarking in CapsNet/AlexNet/W_fc3_em.npy', x)
# plt.hist(x, bins=50, range=(-0.5, 0.5), color='b', edgecolor='grey')
# plt.xlabel('Parameter values')
# plt.ylabel('Numbers')
# plt.show()


# https://gist.githubusercontent.com/saitodev/c4c7a8c83f5aa4a00e93084dd3f848c5/raw/8edd2be3a6382bf589c2b073bc803877280e90db/deep_MNIST_for_experts.py
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Define variables
INPUT_SIZE = 28*28
OUTPUT_NODES = 10
CONV_1_SIZE = 32
CONV_2_SIZE = 64
FC_SIZE = 1024
# Input layer
x  = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='x')
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODES],  name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolutional layer 1
W_conv1 = weight_variable([5, 5, 1, CONV_1_SIZE])
b_conv1 = bias_variable([CONV_1_SIZE])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = weight_variable([5, 5, CONV_1_SIZE, CONV_2_SIZE])
b_conv2 = bias_variable([CONV_2_SIZE])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*CONV_2_SIZE])

W_fc1 = weight_variable([7 * 7 * CONV_2_SIZE, FC_SIZE])
b_fc1 = bias_variable([FC_SIZE])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([FC_SIZE, OUTPUT_NODES])
b_fc2 = bias_variable([OUTPUT_NODES])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  max_steps = 1000
  for step in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(50)

    # Run the network
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    # Test every 100 steps
    if (step % 100) == 0:
      accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
      print(step, accuracy)
  print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
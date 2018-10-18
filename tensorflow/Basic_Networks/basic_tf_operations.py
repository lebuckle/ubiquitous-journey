'''
Created on 9 Aug 2018

@author: Leonie
'''
# https://leimao.github.io/article/Siamese-Network-MNIST/
import tensorflow as tf
import numpy as np
# first, create a TensorFlow constant
const = tf.constant(2.0, name="const")
with tf.device('/cpu:0'):    
	# create TensorFlow variables
	# create TensorFlow variables
	b = tf.placeholder(tf.float32, [None, 1], name='b')
	c = tf.Variable(1.0, name='c')

	# now create some operations
	d = tf.add(b, c, name='d')
	e = tf.add(c, const, name='e')
	a = tf.multiply(d, e, name='a')

# setup the variable initialisation
init_op = tf.global_variables_initializer()

# start the session
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))
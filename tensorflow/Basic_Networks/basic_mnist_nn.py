import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

INPUT_SIZE = 28*28
NO_CLASSES = 10
HIDDEN_LAYER_NODES = 500

checkpoints_dir = 'mymodel'
if os.path.isdir( checkpoints_dir ):
		a = 5
else:
		os.makedirs( checkpoints_dir )

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, NO_CLASSES])

# There are always L-1 number of weights/bias tensors, where L is the number of layers.
# now declare the weights connecting the input to the hidden layer
[input_size*hidden_layer_nodes]
W1 = tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_LAYER_NODES], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([HIDDEN_LAYER_NODES]), name='b1')

# W3 = tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_LAYER_NODES], stddev=0.03), name='W1')
# b3 = tf.Variable(tf.random_normal([HIDDEN_LAYER_NODES]), name='b1')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([HIDDEN_LAYER_NODES, NO_CLASSES], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([NO_CLASSES]), name='b2')

# calculate the output of the hidden layer
# output_layer1 = (input*weights1) + bias1
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
# output = (hidden_out * W2) + b2
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

# Ensure that y is a valid value
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                 + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# To save the model
saver = tf.train.Saver()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
          avg_cost = 0
          for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                h_out, _, c = sess.run([y_, optimiser, cross_entropy], 
                                 feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
          print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
          print("{} {}".format(h_out.shape, batch_x.shape))
    
    saver.save(sess, os.path.join("mymodel/", "my_model.ckpt"),global_step=epoch)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
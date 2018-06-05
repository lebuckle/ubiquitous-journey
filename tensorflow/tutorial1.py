import tensorflow as tf

# initialise constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# multiply
result = tf.multiply(x1,x2)

result2 = tf.multiply(result,1)
# Initialise the session and run result
with tf.Session() as sess:
	# output = sess.run(result)
	output2 = sess.run(result2)
	# print(output)
	print(output2)

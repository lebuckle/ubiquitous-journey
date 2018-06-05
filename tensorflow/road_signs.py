# Tutorial from https://www.datacamp.com/community/tutorials/tensorflow-tutorial
import tensorflow as tf
import os
import skimage
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import random
# Function to load in the data
def load_data(input_directory):

  directories = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]

  labels = []
  images = []

  # Load in the labels
  # Lables are the names of the folders
  for d in directories:
    label_directory = os.path.join(input_directory, d)
    # Find the filenames
    file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]

    # Create the images and the labels
    for f in file_names:
      images.append(skimage.data.imread(f))
      labels.append(int(d))

  return images, labels

def show_unique_labels(images, labels):
  # Find unique labels
  unique_labels = sorted(set(labels))

  # Initialise figure
  plt.figure(figsize=(15,15))

  # Set counter 
  i = 1

  for label in unique_labels:
    # Pick an image for that label
    image = images[labels.index(label)]

    # Define 64 subplots
    plt.subplot(8,8,i)

    # Don't include axes
    plt.axis('off')

    # Add title
    plt.title("Label {}: {}" .format(label, labels.count(label)))

    # Increment counter
    i = i + 1

    plt.imshow(image)

  plt.show()

def rescale(images):

  # Rescale to 28x28
  images28 = [transform.resize(image, (28,28)) for image in images]
  return images28

def to_grayscale(images):
  # Convert images to an array
  images = np.array(images)

  # Convert to grayscale
  images_gray = rgb2gray(images)

  return images_gray

if __name__ == "__main__":
  print("Starting")

  ROOT_PATH = "./"

  train_data_directory = ROOT_PATH + "Training"
  test_data_directory = ROOT_PATH + "Testing"

  # Load the images and lables
  images, labels = load_data(train_data_directory)

  # show_unique_labels(images, labels)

  # Rescale images
  images28 = rescale(images)

  # Convert to grayscale
  gray_images = to_grayscale(images28)

  print("Gray images shape: {}" .format(gray_images[0].shape))
  # Set up place holders for the data
  x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
  y = tf.placeholder(dtype = tf.int32, shape = [None])

  # Flatten the input images
  flat_images = tf.contrib.layers.flatten(x)

  # Fully connected layer 
  no_labels = 62
  logits = tf.contrib.layers.fully_connected(flat_images, no_labels, tf.nn.relu)

  # logits = tf.contrib.layers.fully_connected(flat_images, no_labels, tf.nn.relu)

  # Define a loss function
  # This computes sparse softmax cross entropy between logits and labels. 
  # In other words, it measures the probability error in discrete classification tasks in which the classes are mutually exclusive. 
  # This means that each entry is in exactly one class. 
  # You wrap this function with reduce_mean(), which computes the mean of elements across dimensions of a tensor.
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))

  # Define an optimizer 
  train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

  # Convert logits to label indexes
  correct_pred = tf.argmax(logits, 1)

  # Define an accuracy metric
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  tf.set_random_seed(1234)

  with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('output/logs', sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(1001):
      # print("i: {}" .format(i))
      _, loss_value = sess.run([train_op, loss], feed_dict={x: gray_images, y: labels})
      if i % 10 == 0:
        print("Loss: ", loss_value)

        summary = tf.Summary()
        summary.value.add(tag='loss', simple_value=loss_value)
        #tf.summary.scalar('learning rate', lr)

        summary_writer.add_summary(summary, i)

    # Evaluate the model

    # Load in test data
    test_images, test_labels = load_data(test_data_directory)

    # Rescale images
    test_images28 = rescale(images)

    # Convert to grayscale
    test_gray_images = to_grayscale(images28)

    # Run predictions against the full test set.
    predicted = sess.run([correct_pred], feed_dict={x: test_gray_images})[0]

    # Calculate correct matches 
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

    # Calculate the accuracy
    accuracy = match_count / len(test_labels)

    # Print the accuracy
    print("Accuracy: {:.3f}".format(accuracy))

    # # Pick 10 random images
    # sample_indexes = random.sample(range(len(gray_images)), 10)
    # sample_images = [gray_images[i] for i in sample_indexes]
    # sample_labels = [labels[i] for i in sample_indexes]

    # print("Sample image shape: {}" .format(sample_images[0].shape))
    # predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

    # print("sample labels: {}" .format(sample_labels))
    # print("predicted: {}" .format(predicted))


    # # Display the predictions and the ground truth visually.
    # fig = plt.figure(figsize=(10, 10))
    # for i in range(len(sample_images)):
    #     truth = sample_labels[i]
    #     prediction = predicted[i]
    #     plt.subplot(5, 2,1+i)
    #     plt.axis('off')
    #     color='green' if truth == prediction else 'red'
    #     plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
    #              fontsize=12, color=color)
    #     plt.imshow(sample_images[i],  cmap="gray")

    # plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf


class SimpleMnistTrainer:
    def __init__(self):
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.number_of_examples = 0
        self.train_images = None
        self.train_labels = None

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        # When all training data have been already used, it is reordered randomly.
        if self.index_in_epoch > self.number_of_examples:
            # finished epoch
            self.epochs_completed += 1

            # Shuffle the data.
            perm = np.arange(self.number_of_examples)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]

            # Start next epoch.
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.number_of_examples

        end = self.index_in_epoch
        return self.train_images[start:end], self.train_labels[start:end]


class Tools:
    @staticmethod
    def display(image, width, height):
        image = image.reshape(width, height)
        plt.imshow(image, cmap=cm.get_cmap("binary"))

    @staticmethod
    def dense_to_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def convolution2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

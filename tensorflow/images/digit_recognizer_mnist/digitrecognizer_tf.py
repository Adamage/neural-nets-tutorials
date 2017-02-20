import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import os
import sys


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
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 1500
DROPOUT = 0.5
BATCH_SIZE = 50
# set to 0 to train on all available data
VALIDATION_SIZE = 2000
# image number to output
IMAGE_TO_DISPLAY = 10
LOCAL_PATH = r'C:\Users\msi\Source\Repos\neural-nets-tutorials\tensorflow\images\digit_recognizer_mnist\data'
TRAIN_DATA = os.path.join(LOCAL_PATH, "train.csv")
TEST_DATA = os.path.join(LOCAL_PATH, "test.csv")

if __name__ == "__main__":
    data = pd.read_csv(TRAIN_DATA)
    print('data({0[0]},{0[1]})'.format(data.shape))
    print(data.head())

    images = data.iloc[:, 1:].values
    images = images.astype(np.float)
    images = np.multiply(images, 1.0 / 255.0)
    image_size = images.shape[1]
    Tools.display(images[33], 28, 28)
    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

    labels_flat = data[[0]].values.ravel()
    labels_count = np.unique(labels_flat).shape[0]
    labels = Tools.dense_to_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]
    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]

    # Define Neural network.
    x = tf.placeholder('float', shape=[None, image_size])
    y_labels = tf.placeholder('float', shape=[None, labels_count])

    # First convolutional layer
    W_conv1 = Tools.weight_variable(shape=[5, 5, 1, 32])
    b_conv1 = Tools.bias_variable([32])
    image = tf.reshape(x, [-1, image_width, image_height, 1])
    h_conv1 = tf.nn.relu(Tools.convolution2d(image, W_conv1) + b_conv1)
    h_pool1 = Tools.max_pool_2x2(h_conv1)

    # Visualization
    # show 32 features in 4 by 8 grid
    layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4, 8))
    # reorder: channels, x, y
    layer1 = tf.transpose(layer1, (0, 3, 1, 4, 2))
    layer1 = tf.reshape(tensor=layer1, shape=(int(image_height) * 4, int(image_width * 8)), name="Layer1Reshape")

    # second convolutional layer
    W_conv2 = Tools.weight_variable([5, 5, 32, 64])
    b_conv2 = Tools.bias_variable([64])
    h_conv2 = tf.nn.relu(Tools.convolution2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = Tools.max_pool_2x2(h_conv2)

    # Prepare for visualization
    # display 64 features in 4 by 16 grid
    layer2 = tf.reshape(h_conv2, (-1, 14, 14, 4, 16))
    # reorder so the channels are in the first dimension, x and y follow.
    layer2 = tf.transpose(layer2, (0, 3, 1, 4, 2))
    layer2 = tf.reshape(layer2, (-1, 14 * 4, 14 * 16))

    # add densely connected layer
    W_fc1 = Tools.weight_variable([7 * 7 * 64, 1024])
    b_fc1 = Tools.bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout to prevent overfitting
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # add softmax layer
    # readout layer for deep net
    W_fc2 = Tools.weight_variable([1024, labels_count])
    b_fc2 = Tools.bias_variable([labels_count])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # cost function
    cross_entropy = -tf.reduce_sum(y_labels * tf.log(y))
    # optimisation function
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    # evaluate
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    predict = tf.argmax(y, 1)

    # visualisation variables
    train_accuracies = []
    validation_accuracies = []
    x_range = []
    display_step = 1

    # Stochastic training.
    epochs_completed = 0
    index_in_epoch = 0
    num_examples = train_images.shape[0]

    saver = tf.train.Saver()
    MODEL_PATH = os.path.join(LOCAL_PATH, "model.ckpt")
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        if os.path.isfile(MODEL_PATH + ".meta"):
            ckpt = tf.train.get_checkpoint_state(LOCAL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
        else:
            for i in range(TRAINING_ITERATIONS):
                batch_xs, batch_ys = next_batch(BATCH_SIZE)
                if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:
                    train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_labels: batch_ys, keep_prob: 1.0})

                    validation_accuracy = accuracy.eval(feed_dict={x: validation_images[0:BATCH_SIZE],
                                                                   y_labels: validation_labels[0:BATCH_SIZE],
                                                                   keep_prob: 1.0})
                    print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (train_accuracy,
                                                                                                  validation_accuracy,
                                                                                                  i))
                    validation_accuracies.append(validation_accuracy)
                    train_accuracies.append(train_accuracy)
                    x_range.append(i)
                    # Increase display_step
                    if i % (display_step * 10) == 0 and i:
                        display_step *= 10

                session.run(train_step, feed_dict={x: batch_xs, y_labels: batch_ys, keep_prob: DROPOUT})

            # check final accuracy on validation set
            validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                           y_labels: validation_labels,
                                                           keep_prob: 1.0})
            print('validation_accuracy => %.4f' % validation_accuracy)
            plt.plot(x_range, train_accuracies, '-b', label='Training')
            plt.plot(x_range, validation_accuracies, '-g', label='Validation')
            plt.legend(loc='lower right', frameon=False)
            plt.ylim(ymax=1.1, ymin=0.7)
            plt.ylabel('accuracy')
            plt.xlabel('step')
            plt.show()
            saver.save(session, MODEL_PATH)

        test_images = pd.read_csv(TEST_DATA).values
        test_images = test_images.astype(np.float)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        predicted_labels = np.zeros(test_images.shape[0])
        for i in range(0, test_images.shape[0] // BATCH_SIZE):
            if i % 10 == 0:
                sys.stdout.write(".")
            predicted_labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = \
                predict.eval(feed_dict={x: test_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE], keep_prob: 1.0})

        print('predicted_labels({0})'.format(len(predicted_labels)))

        # Show test image and prediction
        Tools.display(test_images[IMAGE_TO_DISPLAY], image_width, image_height)
        print('predicted_labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY, predicted_labels[IMAGE_TO_DISPLAY]))
        input('Generated test image 10..press')

        # save results
        # np.savetxt('submission_softmax.csv',
        #            np.c_[range(1, len(test_images) + 1), predicted_labels],
        #            delimiter=',',
        #            header='ImageId,Label',
        #            comments='',
        #            fmt='%d')
        #
        layer1_grid = layer1.eval(feed_dict={x: test_images[IMAGE_TO_DISPLAY:IMAGE_TO_DISPLAY + 1], keep_prob: 1.0})
        plt.imshow(layer1_grid[0])
        input('Generated test image layer1..press')
        layer2_grid = layer2.eval(feed_dict={x: test_images[IMAGE_TO_DISPLAY:IMAGE_TO_DISPLAY + 1], keep_prob: 1.0})
        plt.imshow(layer2_grid[0])
        input('Generated test image layer2..press')


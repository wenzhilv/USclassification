#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  lwz 20180927

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot
from tensorflow.examples.tutorials.mnist import input_data


def mnist_regression():
    study_rate = 0.01
    batch_size = 100

    mymnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(dtype="float", shape=[None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W)+b)
    y_ = tf.placeholder(dtype="float",shape=[None, 10])

    # the core
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(study_rate).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    '''
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            batch_xs, batch_ys = mymnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    '''
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mymnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={x: mymnist.test.images, y_: mymnist.test.labels}))
    sess.close()

def weight_variable(theshape):
    initial = tf.truncated_normal(shape=theshape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(theshape):
    initial = tf.constant(0.1, shape=theshape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def mnist_cnn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # convlution layer 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # convlution layer 2
    # error #
    # W_conv2 = weight_variable([5, 5, 32, 64])
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(r'..\log', sess.graph)
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}, session=sess)
            print("step %d, training accuracy %g" % (i, train_accuracy), train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, session=sess))
    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}, session=sess))

    sess.close()


if __name__ == '__main__':
    print("start the regression !\n")
    # mnist_regression()
    print("end the regression !\n")

    print("start the cnn !\n")
    mnist_cnn()
    print("end the cnn !\n")

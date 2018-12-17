#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:27:44 2018

@author: shirhe-lyh
"""

"""Train a CNN model to classifying 10 digits.

Example Usage:
---------------
python3 train.py 
    --images_path: Path to the training images (directory).
    --model_output_path: Path to model.ckpt.
"""


import glob
import numpy as np
import os
import tensorflow as tf
import skimage.io
import skimage.transform
import skimage.color
import tfexample.model as model

flags = tf.app.flags
flags.DEFINE_string('images_path', r'H:\lwz\JL\project\cui\AI\img\out256', 'Path to training images.')
flags.DEFINE_string('model_output_path', r'H:\lwz\JL\project\cui\AI\img\log\checkpoint', 'Path to model checkpoint.')
flags.DEFINE_string('tensorboard_dir', r'H:\lwz\JL\project\cui\AI\img\log\tensorboard', 'tensorboard dir.')
FLAGS = flags.FLAGS


def read_img(path, w=256, h=256, depth=1):
    cate = [path + '\\' + x for x in os.listdir(path) if os.path.isdir(path + '\\' + x)]
    imgs = []
    labels = []
    timg = np.empty([w, h, 1], dtype=np.uint8)
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '\\*.png'):
            print('reading the images:%s' % (im))
            img = skimage.io.imread(im)
            if 1 == depth and len(img.shape) > 2:
                img = skimage.color.rgb2gray(img)
            elif 3 == depth and len(img.shape) < 3:
                img = skimage.color.gray2rgb(img)
            if img.shape[0] != w or img.shape[1] != h:
                img = skimage.transform.resize(img, (w, h))
            timg[:, :, 0] = img
            imgs.append(timg)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def get_train_data(images_path):
    """Get the training images from images_path.

    Args:
        images_path: Path to trianing images.

    Returns:
        images: A list of images.
        lables: A list of integers representing the classes of images.

    Raises:
        ValueError: If images_path is not exist.
    """
    if not os.path.exists(images_path):
        raise ValueError('images_path is not exist.')

    images = []
    labels = []
    images_path = os.path.join(images_path, '*.jpg')
    count = 0
    for image_file in glob.glob(images_path):
        count += 1
        if count % 100 == 0:
            print('Load {} images.'.format(count))
        image = skimage.io.imread(image_file)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = skimage.color.gray2rgb(image)
        # Assume the name of each image is imagexxx_label.jpg
        label = int(image_file.split('_')[-1].split('.')[0])
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def next_batch_set(images, labels, batch_size=128):
    """Generate a batch training data.

    Args:
        images: A 4-D array representing the training images.
        labels: A 1-D array representing the classes of images.
        batch_size: An integer.

    Return:
        batch_images: A batch of images.
        batch_labels: A batch of labels.
    """
    indices = np.random.choice(len(images), batch_size)
    batch_images = images[indices]
    batch_labels = labels[indices]
    return batch_images, batch_labels


def main1(_):
    inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')

    cls_model = model.Model(is_training=True, num_classes=10)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    classes = postprocessed_dict['classes']
    classes_ = tf.identity(classes, name='classes')
    acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), 'float'))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_step = optimizer.minimize(loss, global_step)

    saver = tf.train.Saver()

    images, targets = get_train_data(FLAGS.images_path)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(6000):
            batch_images, batch_labels = next_batch_set(images, targets)
            train_dict = {inputs: batch_images, labels: batch_labels}

            sess.run(train_step, feed_dict=train_dict)

            loss_, acc_ = sess.run([loss, acc], feed_dict=train_dict)

            train_text = 'step: {}, loss: {}, acc: {}'.format(
                i + 1, loss_, acc_)
            print(train_text)

        saver.save(sess, FLAGS.model_output_path)


def main(_):
    cls_num = 2
    img_w = 256
    img_h = 256
    img_depth = 1
    img_batch = 128
    inputs = tf.placeholder(tf.float32, shape=[None, img_w, img_h, img_depth], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')

    # tf.summary.scalar('inputs', inputs)
    # tf.summary.scalar('labels', labels)
    # images, targets = get_train_data(FLAGS.images_path)
    images, targets = read_img(FLAGS.images_path, w=img_w, h=img_h, depth=img_depth)
    # batch_images, batch_labels = next_batch_set(images, targets, batch_size=img_batch)

    print('create model start')
    '''
    with tf.name_scope('inputs'):
        inputs = batch_images
    with tf.name_scope('inputs'):
        labels = batch_labels
    '''

    cls_model = model.Model(is_training=True, num_classes=cls_num)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict, labels)

    # loss = loss_dict['loss']

    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.losses.sparse_softmax_cross_entropy on the
        # raw logit outputs of the nn_layer above.
        with tf.name_scope('total'):
            loss = loss_dict['loss']
    tf.summary.scalar('cross_entropy', loss)


    postprocessed_dict = cls_model.postprocess(prediction_dict)
    classes = postprocessed_dict['classes']
    classes_ = tf.identity(classes, name='classes')

    # acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), 'float'))
    # acc = tf.reduce_mean(tf.cast(tf.equal(classes_, labels), 'float'))

    with tf.name_scope('accuracy'):
        with tf.name_scope('accuracy'):
            acc = tf.reduce_mean(tf.cast(tf.equal(classes_, labels), 'float'))
    tf.summary.scalar('accuracy', acc)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_step = optimizer.minimize(loss, global_step)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    print('sess start')
    with tf.Session() as sess:
        # merged_summary_op = tf.summary.merge_all()
        merged = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(FLAGS.tensorboard_dir, sess.graph)
        train_writer = tf.summary.FileWriter(FLAGS.tensorboard_dir + '/train', sess.graph)
        sess.run(init)
        print('train start')
        for i in range(1000):
            batch_images, batch_labels = next_batch_set(images, targets, batch_size=img_batch)
            train_dict = {inputs: batch_images, labels: batch_labels}

            # sess.run(train_step, feed_dict=train_dict)
            summary, _ = sess.run([merged, train_step], feed_dict=train_dict)

            loss_, acc_ = sess.run([loss, acc], feed_dict=train_dict)

            train_text = 'step: {}, loss: {}, acc: {}'.format(
                i + 1, loss_, acc_)
            print(train_text)
            if 0 == i % 5:
                train_writer.add_summary(summary, i)
                saver.save(sess, os.path.join(FLAGS.model_output_path, r'model-%d.ckpt' % i))
        saver.save(sess, FLAGS.model_output_path)


if __name__ == '__main__':
    print('start')
    tf.app.run(main)
    print('over')

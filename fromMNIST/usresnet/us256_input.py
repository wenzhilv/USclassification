

""" dataset input module.
"""
import glob
import os

import numpy as np
import matplotlib.image as mptim
import matplotlib.pyplot as mplt
import skimage.io
import skimage.transform
import skimage.color
import tensorflow as tf


def img_read_resize_gray(path,image_width,image_height):
    img = skimage.io.imread(path)
    imgsresize = skimage.transform.resize(img, (image_width, image_height))
    imgray = skimage.color.rgb2gray(imgsresize)
    return imgray


def test_imread2(dir, num_classes, image_width, image_height, imgextension='png', img_depth=1):
    """

    :param dir:
    :param num_classes:
    :param image_width:
    :param image_height:
    :param imgextension:
    :param img_depth:
    :return:resimg -- [imgnum, size, size, depth]
            reslabel2 -- [imgnum, num_classes]
    """
    clasindex = []
    cntimg = 0
    for index in range(num_classes):
        filepath = os.path.join(dir, str(index), '*.' + imgextension)
        labelfiles = glob.glob(filepath)
        clasindex.append(len(labelfiles))
        cntimg += len(labelfiles)
    resimg = np.empty([cntimg, image_width,image_height], dtype=np.uint8)
    reslabel = np.zeros([cntimg, 1], dtype=np.uint8)

    cntimg = 0
    for index in range(num_classes):
        filepath = os.path.join(dir, str(index), '*.'+imgextension)
        labelfiles = glob.glob(filepath)
        for file in labelfiles:
            img_gray = img_read_resize_gray(file, image_width, image_height)
            resimg[cntimg, :, :] = img_gray
            reslabel[cntimg, 0] = index
            cntimg += 1
    cntimg = 0
    reslabel2 = np.zeros([cntimg, num_classes], dtype=np.uint8)
    for index in range(cntimg):
        if 0 == reslabel[index]:
            reslabel2[index, 0] = 1
        elif 1 == reslabel[index]:
            reslabel2[index, 1] = 1
        else:
            raise ValueError('Not supported class %d data', index)
    return resimg, reslabel2


def build_input_usimg(dataset, imgs, labels, batch_size, mode):
    image_size = 32
    depth = 3
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0
        num_classes = 10
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1
        num_classes = 100
    # changed by lwz
    elif dataset == 'usimg':
        image_size = 256
        label_bytes = 1
        label_offset = 0
        num_classes = 2
        # depth = 3
        depth = 1
    else:
        raise ValueError('Not supported dataset %s', dataset)


def build_input(dataset, data_path, batch_size, mode):
    """Build CIFAR image and labels.

    Args:
      dataset: Either 'cifar10' or 'cifar100'. add 'usimg'
      data_path: Filename for data.
      batch_size: Input batch size.
      mode: Either 'train' or 'eval'.
    Returns:
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
    Raises:
      ValueError: when the specified dataset is not supported.
    """
    image_size = 32
    depth = 3
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0
        num_classes = 10
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1
        num_classes = 100
    # changed by lwz
    elif dataset == 'usimg':
        image_size = 128
        label_bytes = 1
        label_offset = 0
        num_classes = 2
        # depth = 3
        depth = 1
    else:
        raise ValueError('Not supported dataset %s', dataset)

    image_bytes = image_size * image_size * depth
    record_bytes = label_bytes + label_offset + image_bytes

    data_files = tf.gfile.Glob(data_path)
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    # Read examples from files in the filename queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)

    # Convert these examples to dense labels and processed images.
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
    # Convert from string to [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record, [label_offset + label_bytes], [image_bytes]),
                             [depth, image_size, image_size])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    if mode == 'train':
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size + 4, image_size + 4)
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
        # image = tf.image.random_brightness(image, max_delta=63. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.RandomShuffleQueue(
            capacity=16 * batch_size,
            min_after_dequeue=8 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 16
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size, image_size)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 1

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_classes], 1.0, 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, labels




def build_inputus(dataset, data_path, batch_size, mode):
    """
    :param dataset:  'usimg'
    :param data_path:  root image path
    :param batch_size: batch_size
    :param mode: 'train' or 'eval'
    :return: images , labels
    """
    image_size = 32
    depth = 3
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0
        num_classes = 10
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1
        num_classes = 100
    # changed by lwz
    elif dataset == 'usimg':
        image_size = 128
        label_bytes = 1
        label_offset = 0
        num_classes = 2
        # depth = 3
        depth = 1
    else:
        raise ValueError('Not supported dataset %s', dataset)
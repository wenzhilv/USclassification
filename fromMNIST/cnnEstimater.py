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
#  lwz 20180918

import random
import numpy as np
import tensorflow as tf
# import tensorboard as tb
import argparse
import skimage.io
import skimage.transform
import skimage.color
import glob
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--train_steps', default=100, type=int,
                    help='number of training steps')


def read_img(path, w=128, h=128, depth=1):
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


def load_data():
    train_x = dict()
    train_y = []  # dict()
    test_x = dict()
    test_y = []  # dict()
    # val = 0
    # key = 0
    myfeaturename = ('feature1', 'feature2', 'feature3')
    mycls = ('windows', 'linux', 'all')
    feature1cld0 = np.random.randint(low=0, high=30, size=50)
    feature1cls1 = np.random.randint(low=30, high=60, size=50)
    feature1cls2 = np.random.randint(low=30, high=60, size=50)

    feature2cls0 = np.random.randint(low=30, high=60, size=50)
    feature2cls1 = np.random.randint(low=60, high=90, size=50)
    feature2cls2 = np.random.randint(low=0, high=30, size=50)

    feature3cls0 = np.random.randint(low=60, high=90, size=50)
    feature3cls1 = np.random.randint(low=0, high=30, size=50)
    feature3cls2 = np.random.randint(low=30, high=60, size=50)

    feature1 = np.append( feature1cld0, feature1cls1)
    feature1 = np.append(feature1, feature1cls2)

    feature2 = np.append( feature2cls0, feature2cls1)
    feature2 = np.append(feature2, feature2cls2)

    feature3 = np.append( feature3cls0, feature3cls1)
    feature3 = np.append(feature3, feature3cls2)

    cls0_index = np.ones([50,1], dtype = np.int64) * 0
    cls1_index = np.ones([50, 1], dtype=np.int64) * 1
    cls2_index = np.ones([50, 1], dtype=np.int64) * 2
    cls_index = np.append(cls0_index, cls1_index)
    cls_index = np.append(cls_index, cls2_index)
    # feature1 feature2 feature3 cls_index
    # myfeaturename mycls
    train_x[myfeaturename[0]] = feature1
    train_x[myfeaturename[1]] = feature2
    train_x[myfeaturename[2]] = feature3
    train_y = cls_index

    feature_test0 = np.random.randint(low=0, high=30, size=20)
    feature_test1 = np.random.randint(low=30, high=60, size=20)
    feature_test2 = np.random.randint(low=60, high=90, size=20)
    test_x[myfeaturename[0]] = feature_test0
    test_x[myfeaturename[1]] = feature_test1
    test_x[myfeaturename[2]] = feature_test2
    test_y = np.ones([20,1], dtype = np.int64) * 0
    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))


    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    #(train_x, train_y), (test_x, test_y) = load_data()
    path = r'H:\lwz\JL\project\cui\AI\img\out256'
    output = r'H:\lwz\JL\project\cui\AI\img\log'
    data, label = read_img(path, 256, 256, 1)
    # 打乱顺序
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    # 将所有数据分为训练集和验证集
    ratio = 0.9
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    num_classes = 2
    one_hot_labels = np.zeros([y_train.shape[0], num_classes], dtype=np.uint16)
    for index in range(y_train.shape[0]):
        if 0 == y_train[index]:
            one_hot_labels[index, 0] = 1
        else:
            one_hot_labels[index, 1] = 1
    #
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(0)):
            # Define the model:
            logits = model(
                images=x_train,
                num_classes=num_classes,
                is_training=True,
                rate=1
            )
            # Specify the loss function:
            add_loss(logits, one_hot_labels, use_rank_loss=True)
            total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('Total_Loss', total_loss)

            # Specify the optimization scheme:
            optimizer = tf.train.AdamOptimizer(0.00003)

            # Set up training.
            train_op = tf.contrib.slim.learning.create_train_op(total_loss, optimizer)

            # Run training.
            tf.contrib.slim.learning.train(
                train_op=train_op,
                logdir=output,
                is_chief=0 == 0,
                number_of_steps=100,
                save_summaries_secs=15,
                save_interval_secs=60
            )


def model(images, num_classes, is_training, rate):
    """Generic model.

  Args:
    images: the input patches, a tensor of size [batch_size, patch_width,
      patch_width, 1].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    rate: Integer, convolution rate. 1 for standard convolution, > 1 for dilated
      convolutions.

  Returns:
    the output logits, a tensor of size [batch_size, 11].

  """
    # Adds a convolutional layer with 32 filters of size [5x5], followed by
    # the default (implicit) Relu activation.
    net = tf.contrib.slim.conv2d(images, 32, [5, 5], padding='SAME', scope='conv1')

    # Adds a [2x2] pooling layer with a stride of 2.
    net = tf.contrib.slim.max_pool2d(net, [2, 2], 2, scope='pool1')

    # Adds a convolutional layer with 64 filters of size [5x5], followed by
    # the default (implicit) Relu activation.
    net = tf.contrib.slim.conv2d(net, 64, [5, 5], padding='SAME', scope='conv2', rate=rate)

    # Adds a [2x2] pooling layer with a stride of 2.
    net = tf.contrib.slim.max_pool2d(net, [2, 2], 2, scope='pool2')

    # Reshapes the hidden units such that instead of 2D maps, they are 1D vectors:
    net = tf.contrib.slim.flatten(net)

    # Adds a fully-connected layer with 1024 hidden units, followed by the default
    # Relu activation.
    net = tf.contrib.slim.fully_connected(net, 1024, scope='fc3')

    # Adds a dropout layer during training.
    net = tf.contrib.slim.dropout(net, 0.5, is_training=is_training, scope='dropout3')

    # Adds a fully connected layer with 'num_classes' outputs. Note
    # that the default Relu activation has been overridden to use no activation.
    net = tf.contrib.slim.fully_connected(net, num_classes, activation_fn=None, scope='fc4')

    return net


def add_loss(logits, one_hot_labels, use_rank_loss=False):
    """Add loss function to tf.losses.

  Args:
    logits: Tensor of logits of shape [batch_size, num_classes]
    one_hot_labels: A `Tensor` of size [batch_size, num_classes], where
      each row has a single element set to one and the rest set to zeros.
    use_rank_loss: Boolean, whether to use rank probability score loss instead
      of cross entropy.
  """
    if not use_rank_loss:
        tf.contrib.slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    else:
        rank_loss = ranked_probability_score(
            tf.nn.softmax(logits), one_hot_labels, dim=1)
        tf.losses.add_loss(tf.reduce_mean(rank_loss))


def ranked_probability_score(predictions, targets, dim, name=None):
    r"""Calculate the Ranked Probability Score (RPS).

  RPS is given by the formula

    sum_{k=1}^K (CDF_{prediction,k} - CDF_{target,k}) ^ 2

  where CDF denotes the emperical CDF and each value of `k` denotes a different
  class, in rank order. The range of possible RPS values is `[0, K - 1]`, where
  `K` is the total number of classes. Perfect predictions have a score of zero.

  This is a better metric than cross-entropy for probabilistic classification of
  ranked targets, because it penalizes wrong guesses more harshly if they
  predict a target that is further away. For deterministic predictions (zero
  or one) ranked probability score is equal to absolute error in the number of
  classes.

  Importantly (like cross entropy), it is a strictly proper score rule: the
  highest expected reward is obtained by predicting the true probability
  distribution.

  For these reasons, it is widely used for evaluating weather forecasts, which
  are a prototypical use case for probabilistic regression.

  References:
    Murphy AH. A Note on the Ranked Probability Score. J. Appl. Meteorol. 1971,
    10:155-156.
    http://dx.doi.org/10.1175/1520-0450(1971)010<0155:ANOTRP>2.0.CO;2

  Args:
    predictions: tf.Tensor with probabilities for each class.
    targets: tf.Tensor with one-hot encoded targets.
    dim: integer dimension which corresponds to different classes in both
      ``predictions`` and ``targets``.
    name: optional string name for the operation.

  Returns:
    tf.Tensor with the ranked probability score.

  Raises:
    ValueError: if predictions and targets do not have the same shape.
  """
    with tf.name_scope(name, 'ranked_probability_score', [predictions,
                                                                  targets]) as scope:
        predictions = tf.convert_to_tensor(predictions, name='predictions')
        targets = tf.convert_to_tensor(targets, name='targets')

        if not predictions.get_shape().is_compatible_with(targets.get_shape()):
            raise ValueError('predictions and targets must have compatible shapes')

        if predictions.dtype.is_floating and targets.dtype.is_integer:
            # it's safe to coerce integer targets to float dtype
            targets = tf.cast(targets, dtype=predictions.dtype)

        cdf_pred = tf.cumsum(predictions, dim)
        cdf_target = tf.cumsum(targets, dim)

        values = (cdf_pred - cdf_target) ** 2

        # If desired, we could add arbitrary weighting in this sum along dim.
        # That would still be a proper scoring rule (it's equivalent to rescaling
        # the discretization):
        # https://www.stat.washington.edu/research/reports/2008/tr533.pdf
        rps = tf.reduce_sum(values, dim, name=scope)

        return rps


if __name__ == '__main__':
    print('sort 3 from lwz')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

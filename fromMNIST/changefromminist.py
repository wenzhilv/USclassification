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
import tensorboard as tb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--train_steps', default=100, type=int,
                    help='number of training steps')


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
    (train_x, train_y), (test_x, test_y) = load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['windows', 'linux', 'all']
    predict_x = {
        'feature1': [10, 40, 70],
        'feature2': [40, 70, 10],
        'feature3': [70, 10, 40],
    }

    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_x, labels=None, batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        SPECIES = ['windows', 'linux', 'all']
        print(template.format(SPECIES[class_id], 100 * probability, expec))


if __name__ == '__main__':
    print('sort 3 from lwz')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

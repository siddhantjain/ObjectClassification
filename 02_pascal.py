from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial

from eval import compute_map
#import models

tf.logging.set_verbosity(tf.logging.INFO)

'''
CLASS_NAMES = [
     'aeroplane',
     'bicycle',
     'bird',
     'boat',
     'bottle',
     'bus',
     'car',
     'cat',
     'chair',
     'cow',
     'diningtable',
     'dog',
     'horse',
     'motorbike',
     'person',
     'pottedplant',
     'sheep',
     'sofa',
     'train',
     'tvmonitor',
]
'''


CLASS_NAMES = [
     'aeroplane'
]


def cnn_model_fn(features, labels, mode, num_classes=20):

    #Siddhanj: TODO Might want to take a random 224x224 crop at train and center 224X224 crop at test time
    if mode == tf.estimator.ModeKeys.PREDICT:
        features["x"] = tf.image.central_crop(features["x"])
    else:
        features["x"] = tf.image.extract_glimpse(features["x"], [224,224], [-1,[0,0]], centered=None, normalized=None, uniform_noise=None, name=None)


    features_flipped = tf.image.random_flip_left_right(features["x"])
    features["x"].append(features_flipped)


    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        strides = 4,
        kernel_size=[11, 11],
        kernel_initializer=tf.initializers.random_normal(0,0.01),
        bias = tf.initializers.zeros(),
        padding="valid",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        strides = 1,
        kernel_initializer=tf.initializers.random_normal(0, 0.01),
        bias=tf.initializers.zeros(),
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    # Convolutional Layer #3,#4,#5 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        strides=1,
        kernel_initializer=tf.initializers.random_normal(0, 0.01),
        bias=tf.initializers.zeros(),
        padding="same",
        activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3, 3],
        strides=1,
        kernel_initializer=tf.initializers.random_normal(0, 0.01),
        bias=tf.initializers.zeros(),
        padding="same",
        activation=None)

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        kernel_initializer=tf.initializers.random_normal(0, 0.01),
        bias=tf.initializers.zeros(),
        padding="same",
        activation=None)

    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
    #TODO: Change the 64*64*64 to the value appropriate for this layer
    pool3_flat = tf.reshape(pool3, [-1, 64 * 64 * 64])

    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096,
                            activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                             activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=num_classes)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.sigmoid(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)

    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   10000, 0.5, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum="0.9")

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def load_pascal(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
    """

    imageLabelDict = {}
    num_images = sum(1 for line in open(data_dir + "/ImageSets/Main/" + split + ".txt"))
    images = np.ndarray(shape=(num_images, 256, 256, 3), dtype=np.float32)
    labels = np.ndarray(shape=(num_images, 20), dtype=np.int32)
    weights = np.ndarray(shape=(num_images, 20), dtype=np.int32)
    weights.fill(1)
    labels.fill(0)
    i=0
    for eachClass in CLASS_NAMES:
        print(eachClass)
        labelNumber = CLASS_NAMES.index(eachClass)
        referenceFile = eachClass + "_" + split +".txt"

        with open(data_dir+"/ImageSets/Main/"+referenceFile) as f:

            for line in f:
                words = line.split()


                if(words[1] == '-1'):
                    continue
                if words[0] in imageLabelDict.keys():
                    labels[imageLabelDict[words[0]], labelNumber] = 1
                    if words[1] == '0':
                        weights[imageLabelDict[words[0]], labelNumber] = 0
                    continue

                imageLabelDict[words[0]] = i
                imageFileName = data_dir+"/JPEGImages/"+words[0]+".jpg"

                rawImage = Image.open(imageFileName)
                rawImage = rawImage.resize([256,256])
                finalImageData = np.array(rawImage.getdata(),np.float32).reshape(rawImage.size[1], rawImage.size[0], 3)
                images[i] = finalImageData
                labels[i,labelNumber] =1
                if words[1] == '0':
                    weights[i, labelNumber] = 0
                i=i+1

    return images,labels, weights


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


def main():
    BATCH_SIZE = 100
    NUM_ITERS = 10000
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="/tmp/pascal_model_scratch")

    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    mAPEstimates = []
    for NUM_ITERS in range(100):
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=10,
            hooks=[logging_hook])
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "w": eval_weights},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        rand_AP = compute_map(
            eval_labels, np.random.random(eval_labels.shape),
            eval_weights, average=None)
        print('Random AP: {} mAP'.format(np.mean(rand_AP)))
        gt_AP = compute_map(
            eval_labels, eval_labels, eval_weights, average=None)
        print('GT AP: {} mAP'.format(np.mean(gt_AP)))
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        print('Obtained {} mAP'.format(np.mean(AP)))
        print('per class:')
        for cid, cname in enumerate(CLASS_NAMES):
            print('{}: {}'.format(cname, _get_el(AP, cid)))
        mAPEstimates.append(np.mean(AP))


    #taking a short cut for this question and instead of figuring out tensorboard, just writing mAP values to file,
    # which I will scp and create a graph using Excel (takes much lesser time than figuring tensorboard right now

    mapEstimatesFile = open('mapEstimates.txt', 'w')
    for item in mAPEstimates:
        mapEstimatesFile.write("%s\n" % item)


if __name__ == "__main__":
    main()

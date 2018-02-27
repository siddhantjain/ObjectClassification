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
'''

def cnn_model_fn(features, labels, mode, num_classes=20):

    if mode == tf.estimator.ModeKeys.PREDICT:
        features["x"] = tf.image.resize_image_with_crop_or_pad(features["x"],224,224)
    else:
        augmentedData = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), features["x"])
        augmentedData = tf.map_fn(lambda img: tf.random_crop(img, [224, 224, 3]), augmentedData)
        features["x"] = augmentedData


    input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])

    # Convolutional Layer #1
    conv1_1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        strides = 1,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2 and Pooling Layer #2
    conv1_2 = tf.layers.conv2d(
        inputs=conv1_1,
        filters=64,
        kernel_size=[3, 3],
        strides = 1,
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3,#4,#5 and Pooling Layer #3
    conv2_1 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    conv2_2 = tf.layers.conv2d(
        inputs=conv2_1,
        filters=128,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

    conv3_1 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    conv3_2 = tf.layers.conv2d(
        inputs=conv3_1,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    conv3_3 = tf.layers.conv2d(
        inputs=conv3_2,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)

    conv4_1 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    conv4_2 = tf.layers.conv2d(
        inputs=conv4_1,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    conv4_3 = tf.layers.conv2d(
        inputs=conv4_2,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2)

    conv5_1 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    conv5_2 = tf.layers.conv2d(
        inputs=conv5_1,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    conv5_3 = tf.layers.conv2d(
        inputs=conv5_2,
        filters=512,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu)

    pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)
    #pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])

    fc6 = tf.layers.conv2d(
        inputs=pool5,
        filters=4096,
        kernel_size=[7, 7],
        strides=1,
        padding="valid",
        activation=tf.nn.relu)

    drop6 = tf.layers.dropout(
        inputs=fc6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    fc7 = tf.layers.conv2d(
        inputs=drop6,
        filters=4096,
        kernel_size=[1, 1],
        strides=1,
        padding="valid",
        activation=tf.nn.relu)

    drop7 = tf.layers.dropout(
        inputs=fc7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    drop7_flatten = tf.reshape(drop7,[-1,4096])
    # Logits Layer
    logits = tf.layers.dense(inputs=drop7_flatten, units=num_classes)

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

    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')

    #TODO: CHECK loss formulation and also the prediction probabilities
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        '''
        summary_hook = tf.train.SummarySaverHook(
            400,
            output_dir="/tmp/pascal_model_vgg",
            summary_op=tf.summary.merge_all())
        '''
        
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, tf.train.get_global_step(),
                                                   1000, 0.5, staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])

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


#['conv2d_1/bias:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'conv2d_2/kernel:0' shape=(3, 3, 64, 128) dtype=float32_ref>, <tf.Variable 'conv2d_2/bias:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'conv2d_3/kernel:0' shape=(3, 3, 128, 128) dtype=float32_ref>, <tf.Variable 'conv2d_3/bias:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'conv2d_4/kernel:0' shape=(3, 3, 128, 256) dtype=float32_ref>, <tf.Variable 'conv2d_4/bias:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'conv2d_5/kernel:0' shape=(3, 3, 256, 256) dtype=float32_ref>, <tf.Variable 'conv2d_5/bias:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'conv2d_6/kernel:0' shape=(3, 3, 256, 256) dtype=float32_ref>, <tf.Variable 'conv2d_6/bias:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'conv2d_7/kernel:0' shape=(3, 3, 256, 512) dtype=float32_ref>, <tf.Variable 'conv2d_7/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'conv2d_8/kernel:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'conv2d_8/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'conv2d_9/kernel:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'conv2d_9/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'conv2d_10/kernel:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'conv2d_10/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'conv2d_11/kernel:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'conv2d_11/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'conv2d_12/kernel:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'conv2d_12/bias:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'dense/kernel:0' shape=(25088, 4096) dtype=float32_ref>, <tf.Variable 'dense/bias:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'dense_1/kernel:0' shape=(4096, 4096) dtype=float32_ref>, <tf.Variable 'dense_1/bias:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'dense_2/kernel:0' shape=(4096, 20) dtype=float32_ref>, <tf.Variable 'dense_2/bias:0' shape=(20,) dtype=float32_ref>]


weights_dict = { 'vgg_16/conv1/conv1_1/biases': 'conv2d/bias',
            'vgg_16/conv1/conv1_1/weights': 'conv2d/kernel',
            'vgg_16/conv1/conv1_2/biases': 'conv2d_1/bias',
            'vgg_16/conv1/conv1_2/weights': 'conv2d_1/kernel',
            'vgg_16/conv2/conv2_1/biases': 'conv2d_2/bias',
            'vgg_16/conv2/conv2_1/weights': 'conv2d_2/kernel',
            'vgg_16/conv2/conv2_2/biases': 'conv2d_3/bias',
            'vgg_16/conv2/conv2_2/weights': 'conv2d_3/kernel',
            'vgg_16/conv3/conv3_1/biases': 'conv2d_4/bias',
            'vgg_16/conv3/conv3_1/weights': 'conv2d_4/kernel',
            'vgg_16/conv3/conv3_2/biases': 'conv2d_5/bias',
            'vgg_16/conv3/conv3_2/weights': 'conv2d_5/kernel',
            'vgg_16/conv3/conv3_3/biases': 'conv2d_6/bias',
            'vgg_16/conv3/conv3_3/weights': 'conv2d_6/kernel',
            'vgg_16/conv4/conv4_1/biases': 'conv2d_7/bias',
            'vgg_16/conv4/conv4_1/weights': 'conv2d_7/kernel',
            'vgg_16/conv4/conv4_2/biases': 'conv2d_8/bias',
            'vgg_16/conv4/conv4_2/weights': 'conv2d_8/kernel',
            'vgg_16/conv4/conv4_3/biases': 'conv2d_9/bias',
            'vgg_16/conv4/conv4_3/weights': 'conv2d_9/kernel',
            'vgg_16/conv5/conv5_1/biases': 'conv2d_10/bias',
            'vgg_16/conv5/conv5_1/weights': 'conv2d_10/kernel',
            'vgg_16/conv5/conv5_2/biases': 'conv2d_11/bias',
            'vgg_16/conv5/conv5_2/weights': 'conv2d_11/kernel',
            'vgg_16/conv5/conv5_3/biases': 'conv2d_12/bias',
            'vgg_16/conv5/conv5_3/weights': 'conv2d_12/kernel',
            'vgg_16/fc6/biases': 'conv2d_13/bias',
            'vgg_16/fc6/weights': 'conv2d_13/kernel',
            'vgg_16/fc7/biases': 'conv2d_14/bias',
            'vgg_16/fc7/weights': 'conv2d_14/kernel',}

class PreTrainedHook(tf.train.SessionRunHook):
    def begin(self):
        print(tf.trainable_variables())
        tf.contrib.framework.init_from_checkpoint('vgg_16.ckpt',weights_dict)

def main():
    BATCH_SIZE = 10
    #NUM_ITERS = 10000
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="/tmp/pascal_model_finetuning")

    tensors_to_log = {"loss": "loss"}

    loadWeightHook = PreTrainedHook()
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=400)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    mAPEstimates = []
    for i in range(10):

        if i==0:
            pascal_classifier.train(
                input_fn=train_input_fn,
                steps=400,
                hooks=[logging_hook,loadWeightHook])
        else:
            pascal_classifier.train(
                input_fn=train_input_fn,
                steps=400,
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

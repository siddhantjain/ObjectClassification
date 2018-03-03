# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial

from eval import compute_map
from tensorflow.core.framework import summary_pb2


def visWeight(dataPoint,fileName):
    maxWeight = np.max(dataPoint)
    minWeight = np.min(dataPoint)

    numRows = 8
    numCols = 8

    fig, axes = plt.subplots(numRows, numCols)

    # Note: Took help with Sowmya Munukutla in figuring out the visualising code
    for l, ax in enumerate(axes.flat):
        # get a single filter
        img_r = (dataPoint[:, :, 0, l] - minWeight) / (maxWeight - minWeight) * 256
        img_g = (dataPoint[:, :, 1, l] - minWeight) / (maxWeight - minWeight) * 256
        img_b = (dataPoint[:, :, 2, l] - minWeight) / (maxWeight - minWeight) * 256
        img = np.dstack((img_r, img_g, img_b)).astype(np.uint8)
        # put it on the grid
        ax.imshow(img)
        # time.sleep(50)
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(fileName)

def visWeights():
    #weights = np.load('conv1Filters.npz')
    weights = np.load('conv1Filters_vgg.npz')

    datapoint1 = weights[1,:,:,:,:]
    datapoint2 = weights[5,:,:,:]
    datapoint3 = weights[9,:,:,:]

    visWeight(datapoint1,'datapoint1_vgg.png')
    visWeight(datapoint2, 'datapoint2_vgg.png')
    visWeight(datapoint3, 'datapoint3_vgg.png')


def main():
    visWeights()

if __name__ == "__main__":
    main()
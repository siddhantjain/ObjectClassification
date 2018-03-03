
import numpy as np
import os.path as osp
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import argparse
import seaborn as sns
import sys

#NOTE: I worked with Sowmya Munukutla on this visualisation

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

def scatter(x, colors):

    palette = np.array(sns.color_palette("hls", 20))

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for i in range(20):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(CLASS_NAMES[i]), fontsize=18)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc


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

def computeAverageLabel(labelArray):
    number_of_test_images = labelArray.shape[0]
    output = np.zeros(number_of_test_images)
    number_of_labels = 20
    for i in range(number_of_test_images):
        counter = 0
        cumulative = 0
        for j in range(number_of_labels):
            if labelArray[i][j] == 1:
                counter += 1
                cumulative += j
        if counter == 0:
            output[i] = 0
        else:
            output[i] = int(float(cumulative) / float(counter))
    return output

args = parse_args()
test_feature_vectors = np.load('fc7Features.npz')
index = np.random.permutation(1000)

test_feature_vectors = test_feature_vectors[index]

label_vector = np.load(args.data_dir + '/test_labels.npy')
label_vector = label_vector[index]

b = TSNE(n_components=2).fit_transform(test_feature_vectors)
label = computeAverageLabel(label_vector)



scatter(b, label)
plt.savefig('tsne-generated.png', dpi=120)



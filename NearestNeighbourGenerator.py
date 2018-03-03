import numpy as np
from sklearn.neighbors import KDTree
import random
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import sys


#NOTE: I collaborated with Sowmya Munukutla in Writing this code


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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Find Nearest Neighbours!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def generateImageDict(data_dir,split):
    imageLabelDict = {}
    num_images = sum(1 for line in open(data_dir + "/ImageSets/Main/" + split + ".txt"))
    images = np.ndarray(shape=(num_images, 256, 256, 3), dtype=np.float32)

    i = 0
    for eachClass in CLASS_NAMES:
        print(eachClass)
        referenceFile = eachClass + "_" + split + ".txt"

        with open(data_dir + "/ImageSets/Main/" + referenceFile) as f:

            for line in f:
                words = line.split()

                if (words[1] == '-1'):
                    continue
                if words[0] in imageLabelDict.keys():
                    continue

                imageLabelDict[words[0]] = i
                i = i + 1
    return imageLabelDict


def write_image(index1,index2,data_dir,prefix,imageLabelDict,split):


    images = np.load(data_dir + '/test_images.npy')
    fig, axes = plt.subplots(1, 2, squeeze=False)

    image_file_dir = osp.join(data_dir, 'JPEGImages')
    imageName1 = next((x for x in imageLabelDict if imageLabelDict[x] == index1), None)
    imageName2 = next((x for x in imageLabelDict if imageLabelDict[x] == index2), None)

    image_file1 = osp.join(image_file_dir, imageName1) + '.jpg'
    image1 = Image.open(image_file1)

    image_file2 = osp.join(image_file_dir, imageName2) + '.jpg'
    image2 = Image.open(image_file2)


    axes[0, 0].imshow(image1)
    axes[0, 1].imshow(image2)

    '''
    for image_index, imag in enumerate(images):
        if image_index == index1:
            image_file = osp.join(image_file_dir, image_name) + '.jpg'
            image = Image.open(image_file)
            axes[0, 0].imshow(image)

        if image_index == index2:
            image_file = osp.join(image_file_dir, image_name) + '.jpg'
            image = Image.open(image_file)
            axes[0, 1].imshow(image)
    '''

    plt.savefig(prefix+str(index1)+".png")


def findNearestNeighbours(args,featureVector,id,imageLabelsDict):
    tree = KDTree(featureVector, metric='euclidean')
    number_of_test_images = featureVector.shape[0]
    random_test_samples = random.sample(range(number_of_test_images), 10)

    for value in random_test_samples:
        dist, ind = tree.query([featureVector[value]], k=2)
        for index in ind:
            write_image(index[0],index[1],args.data_dir, id,imageLabelsDict,split='test')



def main():
    args = parse_args()
    fc7FeaturesAlexNet = np.load('fc7Features.npz')
    fc7FeaturesVGG = np.load('fc7Features_vgg.npz')
    pool5FeaturesAlexNet = np.load('pool5Features.npz')
    pool5Features_vgg = np.load('pool5Features_vgg.npz')

    imageLabelsDict = generateImageDict(args.data_dir,'test')

    findNearestNeighbours(args,fc7FeaturesAlexNet,"AlexNetFC7",imageLabelsDict)
    findNearestNeighbours(args, fc7FeaturesAlexNet, "VGGFC7",imageLabelsDict)
    findNearestNeighbours(args, fc7FeaturesAlexNet, "AlexNetPool5",imageLabelsDict)
    findNearestNeighbours(args, fc7FeaturesAlexNet, "VGGPool5",imageLabelsDict)



if __name__ == "__main__":
    main()
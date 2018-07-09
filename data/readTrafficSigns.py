# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import cv2
import csv
import numpy as np
import image_process as impro

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath, mode='test'):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 43 classes
    if mode == 'train':
        print('Reading Traffic Signs training images')
        for c in range(0, 43):
            print '\t/> Read %d' % c
            prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
            gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
            gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
            gtReader.next()  # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                image = cv2.imread(prefix + row[0])
                image = impro.crop_square(image)
                image = cv2.resize(image, (48, 48))
                # image = impro.equalize_intensity(image)
                images.append(image)  # the 1th column is the filename
                label = row[7]
                labels.append(label)  # the 8th column is the label
                # for i in range(3):
                #     img_augment = impro.transform_image(image, 15, 0, 0)
                #     images.append(img_augment)
                #     labels.append(label)
            gtFile.close()
            # add noise image
            # jitter = impro.transform_image(images[-1], 40, 25, 25)
            # images.append(jitter)
            # labels.append(labels[-1])
    else:
        print('Reading Traffic Signs testing images')
        rootpath = rootpath + '/'
        gtFile = open(rootpath + 'GT-final_test.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        gtReader.next()  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            image = cv2.imread(rootpath + row[0])
            image = impro.crop_square(image)
            image = cv2.resize(image, (48, 48))
            # image = impro.equalize_intensity(image)
            images.append(image)  # the 1th column is the filename
            label = row[7]
            labels.append(label)  # the 8th column is the label
        gtFile.close()

    images = np.array(images, np.uint8)
    labels = np.array(labels, np.uint8)
    print('Number images: ', images.shape[0])
    return images, labels

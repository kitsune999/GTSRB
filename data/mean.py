import read_data
import cv2
import numpy

data_set = read_data.read_data_sets('train')

for i in range(10):
    image = data_set.train.images[126+i]
    image = numpy.asarray(image, numpy.uint8)

    print data_set.train.labels[126+i]
    cv2.imshow('aaa', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

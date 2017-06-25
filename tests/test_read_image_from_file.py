import read_data
from matplotlib import pyplot as plt
import cv2
import numpy as np

dataSets = read_data.read_data_sets()

images = dataSets.train.images
labels = dataSets.train.labels

image = images[0]
label = labels[0]

print label

cv2.imshow('aaa', np.array(image))
cv2.waitKey()
cv2.destroyAllWindows()


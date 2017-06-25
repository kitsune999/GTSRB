import cv2
import image_process as impro
from matplotlib import pyplot as plt

image_file = './Final_Test/Images/00163.ppm'

image = cv2.imread(image_file)
image = impro.crop_square(image)
image = cv2.resize(image, (48, 48))
image = impro.equalize_intensity(image)
hist = cv2.calcHist(image, [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []

# loop over the image channels
for (chan, color) in zip(chans, colors):
    # create a histogram for the current channel and
    # concatenate the resulting histograms for each
    # channel
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)

    # plot the histogram
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
    plt.show()

cv2.imshow('aaa', image)
cv2.waitKey()
cv2.destroyAllWindows()
